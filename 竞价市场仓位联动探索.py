from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"


@dataclass(frozen=True)
class ExposurePolicy:
    name: str
    exposure_ratio: Callable[[dict[str, object]], float]


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("auction_backtest", BACKTEST_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载回测模块: {BACKTEST_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_breadth_data(strategy_mod) -> pd.DataFrame:
    local_path = Path(f"{strategy_mod.OUTPUT_STEM}-市场宽度.csv")
    if local_path.exists():
        breadth_df = strategy_mod.read_csv_with_fallback(local_path)
        breadth_df["日期"] = pd.to_datetime(breadth_df["日期"]).dt.normalize()
        return breadth_df
    return strategy_mod.fetch_market_breadth_history()


def create_trade_record(position, trade_date: pd.Timestamp, raw_exit_price: float, exec_exit_price: float, net_proceeds: float) -> dict[str, object]:
    return {
        "股票代码": position.raw_code,
        "基础代码": position.code,
        "股票简称": position.name,
        "买入日期": position.entry_date.strftime("%Y%m%d"),
        "卖出日期": trade_date.strftime("%Y%m%d"),
        "买入原价": round(position.entry_price_raw, 4),
        "买入执行价": round(position.entry_price_exec, 4),
        "卖出原价": round(raw_exit_price, 4),
        "卖出执行价": round(exec_exit_price, 4),
        "信号热度排名": position.signal_rank,
        "分配资金": round(position.allocated_cash, 2),
        "卖出回款": round(net_proceeds, 2),
        "单笔净收益率": round(net_proceeds / position.allocated_cash - 1, 6),
        "持有自然日": (trade_date - position.entry_date).days,
    }


EXPOSURE_POLICIES = [
    ExposurePolicy("基线满仓", lambda _record: 1.0),
    ExposurePolicy(
        "市场60_200到299半仓",
        lambda record: 0.5 if 200 <= float(record.get("市场60日高低差", -999999)) < 300 else 1.0,
    ),
    ExposurePolicy(
        "市场60_200到399半仓",
        lambda record: 0.5 if 200 <= float(record.get("市场60日高低差", -999999)) < 400 else 1.0,
    ),
    ExposurePolicy(
        "市场60_200到299空仓",
        lambda record: 0.0 if 200 <= float(record.get("市场60日高低差", -999999)) < 300 else 1.0,
    ),
    ExposurePolicy(
        "市场60_300以上半仓",
        lambda record: 0.5 if float(record.get("市场60日高低差", -999999)) >= 300 else 1.0,
    ),
    ExposurePolicy(
        "市场60_200到399七成仓",
        lambda record: 0.7 if 200 <= float(record.get("市场60日高低差", -999999)) < 400 else 1.0,
    ),
]


def run_exposure_policy_backtest(candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame], strategy_mod, policy: ExposurePolicy):
    if candidate_book.empty:
        raise ValueError("候选池为空，无法进行市场仓位联动实验")

    candidate_groups = {
        pd.Timestamp(trade_date): group.sort_values(strategy_mod.BUY_RANK_COLUMN, kind="stable").to_dict("records")
        for trade_date, group in candidate_book.groupby("日期", sort=False)
    }
    trading_calendar = strategy_mod.build_trading_calendar(candidate_book, histories)
    trading_calendar = [trade_date for trade_date in trading_calendar if trade_date >= candidate_book["日期"].min()]

    cash = strategy_mod.INITIAL_CAPITAL
    positions: list[object] = []
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []

    for trade_date in trading_calendar:
        open_slots = strategy_mod.MAX_POSITIONS - len(positions)
        if open_slots > 0 and cash > 1e-8 and trade_date in candidate_groups:
            buyable_records: list[dict[str, object]] = []
            for record in candidate_groups[trade_date]:
                if len(buyable_records) >= open_slots:
                    break
                history_df = histories.get(record["基础代码"], pd.DataFrame())
                quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
                if quote is None or not exact_match:
                    continue
                if pd.isna(record.get("开盘价:不复权今日")) or float(record["开盘价:不复权今日"]) <= 0:
                    continue
                buyable_records.append(record)

            if buyable_records:
                exposure_ratio = max(0.0, min(1.0, policy.exposure_ratio(buyable_records[0])))
                day_budget = cash * exposure_ratio
                if day_budget > 0:
                    budget_per_trade = day_budget / len(buyable_records)
                    if budget_per_trade > 0:
                        for record in buyable_records:
                            position = strategy_mod.create_position(record, budget_per_trade)
                            cash -= budget_per_trade
                            positions.append(position)

        positions_to_close: list[tuple[object, float]] = []
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
            if quote is not None and not pd.isna(quote.get("close")):
                position.last_close = float(quote["close"])
            if trade_date <= position.entry_date or quote is None or not exact_match:
                continue
            prev_close = float(quote.get("prev_close")) if not pd.isna(quote.get("prev_close")) else math.nan
            close_price = float(quote.get("close")) if not pd.isna(quote.get("close")) else math.nan
            if not strategy_mod.is_limit_up_close(position.code, position.name, close_price, prev_close):
                positions_to_close.append((position, close_price))

        for position, close_price in positions_to_close:
            exec_exit_price = close_price * (1 - strategy_mod.SLIPPAGE_RATE)
            net_proceeds = position.shares * exec_exit_price * (
                1 - strategy_mod.SELL_COMMISSION_RATE - strategy_mod.STAMP_DUTY_RATE
            )
            cash += net_proceeds
            trade_records.append(
                create_trade_record(
                    position=position,
                    trade_date=trade_date,
                    raw_exit_price=close_price,
                    exec_exit_price=exec_exit_price,
                    net_proceeds=net_proceeds,
                )
            )
            positions.remove(position)

        position_value = 0.0
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, _ = strategy_mod.get_quote_on_or_before(history_df, trade_date)
            position_value += strategy_mod.estimate_position_value(position, quote)

        equity = cash + position_value
        equity_records.append(
            {
                "日期": trade_date,
                "现金": round(cash, 2),
                "持仓市值": round(position_value, 2),
                "总权益": round(equity, 2),
                "持仓数量": len(positions),
            }
        )

    equity_df = pd.DataFrame(equity_records)
    equity_df["净值"] = equity_df["总权益"] / strategy_mod.INITIAL_CAPITAL
    equity_df["历史高点"] = equity_df["净值"].cummax()
    equity_df["回撤"] = equity_df["净值"] / equity_df["历史高点"] - 1

    trade_df = pd.DataFrame(trade_records)
    max_drawdown = float(equity_df["回撤"].min()) if not equity_df.empty else math.nan
    final_nav = float(equity_df.iloc[-1]["净值"]) if not equity_df.empty else math.nan
    total_return = final_nav - 1 if not math.isnan(final_nav) else math.nan
    summary_df = pd.DataFrame(
        [
            {
                "市场仓位策略": policy.name,
                "期末权益": float(equity_df.iloc[-1]["总权益"]) if not equity_df.empty else math.nan,
                "期末净值": final_nav,
                "总收益率": total_return,
                "最大回撤": max_drawdown,
                "交易笔数": int(len(trade_df)),
                "胜率": float((trade_df["单笔净收益率"] > 0).mean()) if not trade_df.empty else math.nan,
                "平均单笔净收益率": float(trade_df["单笔净收益率"].mean()) if not trade_df.empty else math.nan,
                "收益回撤比": float(total_return / abs(max_drawdown)) if max_drawdown < 0 else math.nan,
                "平均持有自然日": float(trade_df["持有自然日"].mean()) if not trade_df.empty else math.nan,
            }
        ]
    )
    return trade_df, equity_df, summary_df


def export_results(strategy_mod, result_df: pd.DataFrame, best_policy_name: str, best_trade_df: pd.DataFrame, best_equity_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-市场仓位联动实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")
    best_summary = result_df[result_df["市场仓位策略"] == best_policy_name].copy()
    best_summary.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")

    export_equity_df = best_equity_df.copy()
    export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
    best_trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
    export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
    strategy_mod.save_plot(best_equity_df, Path(f"{output_prefix}-最佳策略净值.png"))


def main() -> None:
    strategy_mod = load_strategy_module()
    signal_df = strategy_mod.load_factor_signal_data(strategy_mod.FACTOR_INPUT_CSV)
    breadth_df = load_breadth_data(strategy_mod)
    base = strategy_mod.AUCTION_RATIO_RANK_CONFIG
    candidate_book = strategy_mod.build_candidate_book_with_config(
        signal_df=signal_df,
        breadth_df=breadth_df,
        buy_filters=base.buy_filters,
        named_conditions=base.named_conditions,
        market_filter=base.market_filter,
        sort_by=base.sort_by,
        max_positions=strategy_mod.MAX_POSITIONS,
    )

    all_codes = sorted(candidate_book["基础代码"].dropna().astype(str).unique().tolist())
    histories = strategy_mod.fetch_histories(all_codes)

    summary_frames: list[pd.DataFrame] = []
    best_policy_name = ""
    best_trade_df = pd.DataFrame()
    best_equity_df = pd.DataFrame()
    best_score = -math.inf

    for policy in EXPOSURE_POLICIES:
        trade_df, equity_df, summary_df = run_exposure_policy_backtest(candidate_book, histories, strategy_mod, policy)
        summary_frames.append(summary_df)
        score = float(summary_df.iloc[0]["收益回撤比"])
        if math.isnan(score):
            score = -math.inf
        if score > best_score:
            best_score = score
            best_policy_name = policy.name
            best_trade_df = trade_df
            best_equity_df = equity_df

    result_df = pd.concat(summary_frames, ignore_index=True).sort_values(["收益回撤比", "总收益率"], ascending=[False, False])
    export_results(strategy_mod, result_df, best_policy_name, best_trade_df, best_equity_df)

    print("市场仓位联动实验完成")
    print(result_df.to_string(index=False))
    print(f"\n当前最佳市场仓位策略: {best_policy_name}")


if __name__ == "__main__":
    main()
