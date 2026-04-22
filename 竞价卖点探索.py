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
class ExitPolicy:
    name: str
    open_exit_rule: Callable[[object, pd.Series, int], str | None]
    close_hold_rule: Callable[[object, pd.Series, int], str | None]


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


def to_float(value) -> float:
    if pd.isna(value):
        return math.nan
    return float(value)


def quote_metrics(quote: pd.Series) -> dict[str, float]:
    open_price = to_float(quote.get("open"))
    high_price = to_float(quote.get("high"))
    low_price = to_float(quote.get("low"))
    close_price = to_float(quote.get("close"))
    prev_close = to_float(quote.get("prev_close"))
    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "prev_close": prev_close,
    }


def no_open_exit(_position, _quote: pd.Series, _hold_days: int) -> str | None:
    return None


def weak_open_exit(_position, quote: pd.Series, _hold_days: int) -> str | None:
    metrics = quote_metrics(quote)
    if math.isnan(metrics["open"]) or math.isnan(metrics["prev_close"]) or metrics["prev_close"] <= 0:
        return None
    if metrics["open"] < metrics["prev_close"]:
        return "弱开开盘走"
    return None


def baseline_close_sell(_position, _quote: pd.Series, _hold_days: int) -> str | None:
    return None


def strong_close_hold_3pct(_position, quote: pd.Series, _hold_days: int) -> str | None:
    metrics = quote_metrics(quote)
    if (
        math.isnan(metrics["open"])
        or math.isnan(metrics["close"])
        or math.isnan(metrics["prev_close"])
        or metrics["prev_close"] <= 0
    ):
        return None
    if metrics["close"] >= metrics["prev_close"] * 1.03 and metrics["close"] >= metrics["open"]:
        return "强收继续持有"
    return None


def near_high_hold(_position, quote: pd.Series, _hold_days: int) -> str | None:
    metrics = quote_metrics(quote)
    if (
        math.isnan(metrics["close"])
        or math.isnan(metrics["high"])
        or math.isnan(metrics["prev_close"])
        or metrics["prev_close"] <= 0
        or metrics["high"] <= 0
    ):
        return None
    if metrics["close"] >= metrics["prev_close"] * 1.02 and metrics["close"] >= metrics["high"] * 0.98:
        return "近高强势继续持有"
    return None


EXIT_POLICIES = [
    ExitPolicy("基线", no_open_exit, baseline_close_sell),
    ExitPolicy("弱开先走", weak_open_exit, baseline_close_sell),
    ExitPolicy("强收多拿", no_open_exit, strong_close_hold_3pct),
    ExitPolicy("近高强势多拿", no_open_exit, near_high_hold),
    ExitPolicy("弱开先走+强收多拿", weak_open_exit, strong_close_hold_3pct),
]


def create_trade_record(
    position,
    trade_date: pd.Timestamp,
    raw_exit_price: float,
    exec_exit_price: float,
    net_proceeds: float,
    quote: pd.Series,
    exit_mode: str,
    exit_reason: str,
) -> dict[str, object]:
    metrics = quote_metrics(quote)
    prev_close = metrics["prev_close"]
    open_change = math.nan
    close_change = math.nan
    if not math.isnan(prev_close) and prev_close > 0:
        if not math.isnan(metrics["open"]):
            open_change = metrics["open"] / prev_close - 1
        if not math.isnan(metrics["close"]):
            close_change = metrics["close"] / prev_close - 1

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
        "卖出方式": exit_mode,
        "卖出原因": exit_reason,
        "信号热度排名": position.signal_rank,
        "分配资金": round(position.allocated_cash, 2),
        "卖出回款": round(net_proceeds, 2),
        "单笔净收益率": round(net_proceeds / position.allocated_cash - 1, 6),
        "持有自然日": (trade_date - position.entry_date).days,
        "卖出日开盘": round(metrics["open"], 4) if not math.isnan(metrics["open"]) else math.nan,
        "卖出日最高": round(metrics["high"], 4) if not math.isnan(metrics["high"]) else math.nan,
        "卖出日最低": round(metrics["low"], 4) if not math.isnan(metrics["low"]) else math.nan,
        "卖出日收盘": round(metrics["close"], 4) if not math.isnan(metrics["close"]) else math.nan,
        "卖出日前收": round(prev_close, 4) if not math.isnan(prev_close) else math.nan,
        "卖出日开盘涨跌幅": round(open_change, 6) if not math.isnan(open_change) else math.nan,
        "卖出日收盘涨跌幅": round(close_change, 6) if not math.isnan(close_change) else math.nan,
    }


def build_summary(policy_name: str, trade_df: pd.DataFrame, equity_df: pd.DataFrame, open_exit_count: int) -> pd.DataFrame:
    max_drawdown = float(equity_df["回撤"].min()) if not equity_df.empty else math.nan
    final_nav = float(equity_df.iloc[-1]["净值"]) if not equity_df.empty else math.nan
    final_equity = float(equity_df.iloc[-1]["总权益"]) if not equity_df.empty else math.nan
    total_return = final_nav - 1 if not math.isnan(final_nav) else math.nan
    summary = {
        "卖点策略": policy_name,
        "期末权益": final_equity,
        "期末净值": final_nav,
        "总收益率": total_return,
        "最大回撤": max_drawdown,
        "交易笔数": int(len(trade_df)),
        "胜率": float((trade_df["单笔净收益率"] > 0).mean()) if not trade_df.empty else math.nan,
        "平均单笔净收益率": float(trade_df["单笔净收益率"].mean()) if not trade_df.empty else math.nan,
        "收益回撤比": float(total_return / abs(max_drawdown)) if max_drawdown < 0 else math.nan,
        "平均持有自然日": float(trade_df["持有自然日"].mean()) if not trade_df.empty else math.nan,
        "1日交易占比": float((trade_df["持有自然日"] == 1).mean()) if not trade_df.empty else math.nan,
        "开盘卖出笔数": int(open_exit_count),
        "开盘卖出占比": float((trade_df["卖出方式"] == "开盘").mean()) if not trade_df.empty else math.nan,
    }
    return pd.DataFrame([summary])


def run_exit_policy_backtest(candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame], strategy_mod, policy: ExitPolicy):
    if candidate_book.empty:
        raise ValueError("候选池为空，无法进行卖点实验")

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
    open_exit_count = 0

    for trade_date in trading_calendar:
        open_exits: list[tuple[object, pd.Series, str]] = []
        for position in list(positions):
            history_df = histories.get(position.code, pd.DataFrame())
            quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
            if trade_date <= position.entry_date or quote is None or not exact_match:
                continue
            hold_days = (trade_date - position.entry_date).days
            open_exit_reason = policy.open_exit_rule(position, quote, hold_days)
            if open_exit_reason:
                metrics = quote_metrics(quote)
                if math.isnan(metrics["open"]) or metrics["open"] <= 0:
                    continue
                open_exits.append((position, quote, open_exit_reason))

        for position, quote, exit_reason in open_exits:
            raw_exit_price = float(quote["open"])
            exec_exit_price = raw_exit_price * (1 - strategy_mod.SLIPPAGE_RATE)
            net_proceeds = position.shares * exec_exit_price * (
                1 - strategy_mod.SELL_COMMISSION_RATE - strategy_mod.STAMP_DUTY_RATE
            )
            cash += net_proceeds
            open_exit_count += 1
            trade_records.append(
                create_trade_record(
                    position=position,
                    trade_date=trade_date,
                    raw_exit_price=raw_exit_price,
                    exec_exit_price=exec_exit_price,
                    net_proceeds=net_proceeds,
                    quote=quote,
                    exit_mode="开盘",
                    exit_reason=exit_reason,
                )
            )
            positions.remove(position)

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
                budget_per_trade = cash / len(buyable_records)
                if budget_per_trade > 0:
                    for record in buyable_records:
                        position = strategy_mod.create_position(record, budget_per_trade)
                        cash -= budget_per_trade
                        positions.append(position)

        close_exits: list[tuple[object, pd.Series, str]] = []
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
            if quote is not None and not pd.isna(quote.get("close")):
                position.last_close = float(quote["close"])
            if trade_date <= position.entry_date or quote is None or not exact_match:
                continue

            metrics = quote_metrics(quote)
            prev_close = metrics["prev_close"]
            close_price = metrics["close"]
            if strategy_mod.is_limit_up_close(position.code, position.name, close_price, prev_close):
                continue

            hold_days = (trade_date - position.entry_date).days
            hold_reason = policy.close_hold_rule(position, quote, hold_days)
            if hold_reason is None:
                close_exits.append((position, quote, "非涨停收盘卖出"))

        for position, quote, exit_reason in close_exits:
            raw_exit_price = float(quote["close"])
            exec_exit_price = raw_exit_price * (1 - strategy_mod.SLIPPAGE_RATE)
            net_proceeds = position.shares * exec_exit_price * (
                1 - strategy_mod.SELL_COMMISSION_RATE - strategy_mod.STAMP_DUTY_RATE
            )
            cash += net_proceeds
            trade_records.append(
                create_trade_record(
                    position=position,
                    trade_date=trade_date,
                    raw_exit_price=raw_exit_price,
                    exec_exit_price=exec_exit_price,
                    net_proceeds=net_proceeds,
                    quote=quote,
                    exit_mode="收盘",
                    exit_reason=exit_reason,
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
    summary_df = build_summary(policy.name, trade_df, equity_df, open_exit_count)
    return trade_df, equity_df, summary_df


def export_experiment_results(strategy_mod, all_summaries: pd.DataFrame, best_policy_name: str, best_trade_df: pd.DataFrame, best_equity_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-卖点实验"
    all_summaries.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")
    best_summary = all_summaries[all_summaries["卖点策略"] == best_policy_name].copy()
    best_summary.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")

    export_equity_df = best_equity_df.copy()
    export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
    best_trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
    export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
    strategy_mod.save_plot(best_equity_df, Path(f"{output_prefix}-最佳策略净值.png"))


def main() -> None:
    strategy_mod = load_strategy_module()
    if not strategy_mod.FACTOR_INPUT_CSV.exists():
        raise FileNotFoundError(f"未找到扩展因子文件: {strategy_mod.FACTOR_INPUT_CSV}")

    signal_df = strategy_mod.load_factor_signal_data(strategy_mod.FACTOR_INPUT_CSV)
    breadth_df = load_breadth_data(strategy_mod)
    config = strategy_mod.AUCTION_RATIO_RANK_CONFIG
    candidate_book = strategy_mod.build_candidate_book_with_config(
        signal_df=signal_df,
        breadth_df=breadth_df,
        buy_filters=config.buy_filters,
        named_conditions=config.named_conditions,
        market_filter=config.market_filter,
        sort_by=config.sort_by,
        max_positions=strategy_mod.MAX_POSITIONS,
    )
    if candidate_book.empty:
        raise ValueError("最优买点配置下候选池为空")

    codes = sorted(candidate_book["基础代码"].dropna().astype(str).unique().tolist())
    histories = strategy_mod.fetch_histories(codes)

    summary_frames: list[pd.DataFrame] = []
    best_policy_name = ""
    best_trade_df = pd.DataFrame()
    best_equity_df = pd.DataFrame()
    best_score = -math.inf

    for policy in EXIT_POLICIES:
        trade_df, equity_df, summary_df = run_exit_policy_backtest(candidate_book, histories, strategy_mod, policy)
        summary_frames.append(summary_df)
        score = float(summary_df.iloc[0]["收益回撤比"])
        if math.isnan(score):
            score = -math.inf
        if score > best_score:
            best_score = score
            best_policy_name = policy.name
            best_trade_df = trade_df
            best_equity_df = equity_df

    all_summaries = pd.concat(summary_frames, ignore_index=True)
    all_summaries = all_summaries.sort_values(["收益回撤比", "总收益率", "最大回撤"], ascending=[False, False, False])
    export_experiment_results(strategy_mod, all_summaries, best_policy_name, best_trade_df, best_equity_df)

    print("卖点实验完成")
    print(f"基于买点策略: {config.name}")
    print(all_summaries.to_string(index=False))
    print(f"\n当前最佳卖点策略: {best_policy_name}")


if __name__ == "__main__":
    main()
