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
class SizingPolicy:
    name: str
    allocator: Callable[[list[dict[str, object]], float], list[tuple[dict[str, object], float]]]


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


def numeric_score(record: dict[str, object], column: str) -> float:
    value = pd.to_numeric(pd.Series([record.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return math.nan
    return float(value)


def allocate_equal(records: list[dict[str, object]], cash: float) -> list[tuple[dict[str, object], float]]:
    if not records:
        return []
    budget = cash / len(records)
    return [(record, budget) for record in records if budget > 0]


def allocate_fixed_by_column(
    records: list[dict[str, object]],
    cash: float,
    column: str,
    descending: bool,
    weights: tuple[float, ...],
) -> list[tuple[dict[str, object], float]]:
    if not records:
        return []

    ranked_records = sorted(
        records,
        key=lambda record: (
            math.inf
            if math.isnan(numeric_score(record, column))
            else -numeric_score(record, column)
            if descending
            else numeric_score(record, column)
        ),
    )
    use_weights = list(weights[: len(ranked_records)])
    if len(use_weights) < len(ranked_records):
        use_weights.extend([use_weights[-1]] * (len(ranked_records) - len(use_weights)))
    total_weight = sum(use_weights)
    if total_weight <= 0:
        return allocate_equal(records, cash)
    return [(record, cash * weight / total_weight) for record, weight in zip(ranked_records, use_weights) if weight > 0]


def allocate_sqrt_ratio(records: list[dict[str, object]], cash: float) -> list[tuple[dict[str, object], float]]:
    if not records:
        return []

    ranked_records = sorted(
        records,
        key=lambda record: numeric_score(record, "竞昨成交比估算") if not math.isnan(numeric_score(record, "竞昨成交比估算")) else -math.inf,
        reverse=True,
    )
    weights = []
    for record in ranked_records:
        score = numeric_score(record, "竞昨成交比估算")
        weights.append(math.sqrt(score) if not math.isnan(score) and score > 0 else 0.0)
    total_weight = sum(weights)
    if total_weight <= 0:
        return allocate_equal(records, cash)
    return [(record, cash * weight / total_weight) for record, weight in zip(ranked_records, weights) if weight > 0]


SIZING_POLICIES = [
    SizingPolicy("等权基线", allocate_equal),
    SizingPolicy(
        "竞昨比50_30_20",
        lambda records, cash: allocate_fixed_by_column(
            records,
            cash,
            column="竞昨成交比估算",
            descending=True,
            weights=(0.5, 0.3, 0.2),
        ),
    ),
    SizingPolicy(
        "竞昨比60_30_10",
        lambda records, cash: allocate_fixed_by_column(
            records,
            cash,
            column="竞昨成交比估算",
            descending=True,
            weights=(0.6, 0.3, 0.1),
        ),
    ),
    SizingPolicy("竞昨比平方根比例", allocate_sqrt_ratio),
    SizingPolicy(
        "热度50_30_20",
        lambda records, cash: allocate_fixed_by_column(
            records,
            cash,
            column="个股热度排名昨日",
            descending=False,
            weights=(0.5, 0.3, 0.2),
        ),
    ),
]


def scan_ratio_weight_grid(candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame], strategy_mod) -> pd.DataFrame:
    scan_rows: list[dict[str, object]] = []
    steps = [i / 20 for i in range(1, 20)]
    for first_weight in steps:
        for second_weight in steps:
            third_weight = 1 - first_weight - second_weight
            if third_weight <= 0:
                continue
            if not (first_weight >= second_weight >= third_weight):
                continue

            policy = SizingPolicy(
                name=f"竞昨比权重_{first_weight:.2f}_{second_weight:.2f}_{third_weight:.2f}",
                allocator=lambda records, cash, weights=(first_weight, second_weight, third_weight): allocate_fixed_by_column(
                    records,
                    cash,
                    column="竞昨成交比估算",
                    descending=True,
                    weights=weights,
                ),
            )
            _, _, summary_df = run_sizing_policy_backtest(candidate_book, histories, strategy_mod, policy)
            row = summary_df.iloc[0].to_dict()
            row["权重1"] = first_weight
            row["权重2"] = second_weight
            row["权重3"] = third_weight
            scan_rows.append(row)

    scan_df = pd.DataFrame(scan_rows)
    return scan_df.sort_values(["总收益率", "收益回撤比"], ascending=[False, False]).reset_index(drop=True)


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


def build_summary(policy_name: str, trade_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    max_drawdown = float(equity_df["回撤"].min()) if not equity_df.empty else math.nan
    final_nav = float(equity_df.iloc[-1]["净值"]) if not equity_df.empty else math.nan
    final_equity = float(equity_df.iloc[-1]["总权益"]) if not equity_df.empty else math.nan
    total_return = final_nav - 1 if not math.isnan(final_nav) else math.nan
    summary = {
        "分仓策略": policy_name,
        "期末权益": final_equity,
        "期末净值": final_nav,
        "总收益率": total_return,
        "最大回撤": max_drawdown,
        "交易笔数": int(len(trade_df)),
        "胜率": float((trade_df["单笔净收益率"] > 0).mean()) if not trade_df.empty else math.nan,
        "平均单笔净收益率": float(trade_df["单笔净收益率"].mean()) if not trade_df.empty else math.nan,
        "收益回撤比": float(total_return / abs(max_drawdown)) if max_drawdown < 0 else math.nan,
        "平均持有自然日": float(trade_df["持有自然日"].mean()) if not trade_df.empty else math.nan,
        "平均单笔分配资金": float(trade_df["分配资金"].mean()) if not trade_df.empty else math.nan,
    }
    return pd.DataFrame([summary])


def run_sizing_policy_backtest(candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame], strategy_mod, policy: SizingPolicy):
    if candidate_book.empty:
        raise ValueError("候选池为空，无法进行分仓实验")

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

            allocations = policy.allocator(buyable_records, cash)
            for record, budget in allocations:
                if budget <= 0:
                    continue
                position = strategy_mod.create_position(record, budget)
                cash -= budget
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
    summary_df = build_summary(policy.name, trade_df, equity_df)
    return trade_df, equity_df, summary_df


def export_experiment_results(strategy_mod, all_summaries: pd.DataFrame, best_policy_name: str, best_trade_df: pd.DataFrame, best_equity_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-仓位实验"
    all_summaries.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")
    best_summary = all_summaries[all_summaries["分仓策略"] == best_policy_name].copy()
    best_summary.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")

    export_equity_df = best_equity_df.copy()
    export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
    best_trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
    export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
    strategy_mod.save_plot(best_equity_df, Path(f"{output_prefix}-最佳策略净值.png"))


def export_weight_scan(strategy_mod, scan_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-仓位实验"
    scan_df.to_csv(f"{output_prefix}-竞昨比权重扫描.csv", index=False, encoding="utf-8-sig")


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

    for policy in SIZING_POLICIES:
        trade_df, equity_df, summary_df = run_sizing_policy_backtest(candidate_book, histories, strategy_mod, policy)
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

    weight_scan_df = scan_ratio_weight_grid(candidate_book, histories, strategy_mod)
    export_weight_scan(strategy_mod, weight_scan_df)
    top_return_df = weight_scan_df.head(10).copy()
    top_ratio_df = weight_scan_df.sort_values(["收益回撤比", "总收益率"], ascending=[False, False]).head(10).copy()

    print("仓位实验完成")
    print(f"基于买点策略: {config.name}")
    print(all_summaries.to_string(index=False))
    print("\n竞昨比权重扫描-收益前十:")
    print(top_return_df[["分仓策略", "总收益率", "最大回撤", "收益回撤比"]].to_string(index=False))
    print("\n竞昨比权重扫描-收益回撤比前十:")
    print(top_ratio_df[["分仓策略", "总收益率", "最大回撤", "收益回撤比"]].to_string(index=False))
    print(f"\n当前最佳分仓策略: {best_policy_name}")


if __name__ == "__main__":
    main()
