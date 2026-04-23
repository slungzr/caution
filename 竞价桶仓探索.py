from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
TAIL_PROXY_PATH = BASE_DIR / "竞价尾段代理探索.py"

DAILY_BUY_LIMIT = 3
BUCKET_COUNTS = [3, 4, 5, 6, 7, 8, 9, 12]


@dataclass
class Bucket:
    bucket_id: int
    cash: float
    position: object | None = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_best_candidate_book(strategy_mod, tail_mod) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_mod = load_module("bucket_open_sector", BASE_DIR / "竞价行业开盘联动探索.py")
    base_df = tail_mod.prepare_base_df(strategy_mod, open_mod)
    best_config = next(config for config in tail_mod.build_configs(strategy_mod) if config.name == "未匹配占比优先排序")
    _, candidate_book = tail_mod.build_candidate_book(strategy_mod, base_df, best_config)
    return base_df, candidate_book


def create_trade_record(strategy_mod, position, trade_date: pd.Timestamp, raw_exit_price: float, exec_exit_price: float, net_proceeds: float, bucket_id: int) -> dict[str, object]:
    return {
        "桶编号": bucket_id,
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


def build_summary(label: str, trade_df: pd.DataFrame, equity_df: pd.DataFrame, bucket_count: int) -> pd.DataFrame:
    max_drawdown = float(equity_df["回撤"].min()) if not equity_df.empty else math.nan
    final_nav = float(equity_df.iloc[-1]["净值"]) if not equity_df.empty else math.nan
    final_equity = float(equity_df.iloc[-1]["总权益"]) if not equity_df.empty else math.nan
    total_return = final_nav - 1 if not math.isnan(final_nav) else math.nan
    summary = {
        "方案": label,
        "桶数": bucket_count,
        "期末权益": final_equity,
        "期末净值": final_nav,
        "总收益率": total_return,
        "最大回撤": max_drawdown,
        "交易笔数": int(len(trade_df)),
        "胜率": float((trade_df["单笔净收益率"] > 0).mean()) if not trade_df.empty else math.nan,
        "平均单笔净收益率": float(trade_df["单笔净收益率"].mean()) if not trade_df.empty else math.nan,
        "收益回撤比": float(total_return / abs(max_drawdown)) if max_drawdown < 0 else math.nan,
        "平均持有自然日": float(trade_df["持有自然日"].mean()) if not trade_df.empty else math.nan,
        "最大同时持仓桶数": int(equity_df["持仓桶数"].max()) if not equity_df.empty else 0,
    }
    return pd.DataFrame([summary])


def run_bucket_backtest(strategy_mod, candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame], bucket_count: int):
    candidate_groups = {
        pd.Timestamp(trade_date): group.to_dict("records")
        for trade_date, group in candidate_book.groupby("日期", sort=False)
    }
    trading_calendar = strategy_mod.build_trading_calendar(candidate_book, histories)
    trading_calendar = [trade_date for trade_date in trading_calendar if trade_date >= candidate_book["日期"].min()]

    initial_bucket_cash = strategy_mod.INITIAL_CAPITAL / bucket_count
    buckets = [Bucket(bucket_id=index + 1, cash=initial_bucket_cash) for index in range(bucket_count)]
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []

    for trade_date in trading_calendar:
        free_buckets = [bucket for bucket in buckets if bucket.position is None and bucket.cash > 1e-8]
        if free_buckets and trade_date in candidate_groups:
            buyable_records: list[dict[str, object]] = []
            for record in candidate_groups[trade_date]:
                if len(buyable_records) >= min(DAILY_BUY_LIMIT, len(free_buckets)):
                    break
                history_df = histories.get(record["基础代码"], pd.DataFrame())
                quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
                if quote is None or not exact_match:
                    continue
                if pd.isna(record.get("开盘价:不复权今日")) or float(record["开盘价:不复权今日"]) <= 0:
                    continue
                buyable_records.append(record)

            for bucket, record in zip(free_buckets, buyable_records):
                bucket.position = strategy_mod.create_position(record, bucket.cash)
                bucket.cash = 0.0

        positions_to_close: list[tuple[Bucket, float]] = []
        position_value = 0.0
        holding_bucket_count = 0
        for bucket in buckets:
            if bucket.position is None:
                continue
            holding_bucket_count += 1
            history_df = histories.get(bucket.position.code, pd.DataFrame())
            quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
            position_value += strategy_mod.estimate_position_value(bucket.position, quote)
            if trade_date <= bucket.position.entry_date or quote is None or not exact_match:
                continue
            prev_close = float(quote.get("prev_close")) if not pd.isna(quote.get("prev_close")) else math.nan
            close_price = float(quote.get("close")) if not pd.isna(quote.get("close")) else math.nan
            if not strategy_mod.is_limit_up_close(bucket.position.code, bucket.position.name, close_price, prev_close):
                positions_to_close.append((bucket, close_price))

        for bucket, close_price in positions_to_close:
            exec_exit_price = close_price * (1 - strategy_mod.SLIPPAGE_RATE)
            net_proceeds = bucket.position.shares * exec_exit_price * (
                1 - strategy_mod.SELL_COMMISSION_RATE - strategy_mod.STAMP_DUTY_RATE
            )
            bucket.cash = net_proceeds
            trade_records.append(
                create_trade_record(
                    strategy_mod,
                    bucket.position,
                    trade_date,
                    close_price,
                    exec_exit_price,
                    net_proceeds,
                    bucket.bucket_id,
                )
            )
            bucket.position = None

        total_cash = sum(bucket.cash for bucket in buckets)
        total_equity = total_cash + position_value
        equity_records.append(
            {
                "日期": trade_date,
                "总权益": total_equity,
                "净值": total_equity / strategy_mod.INITIAL_CAPITAL,
                "现金": total_cash,
                "持仓市值": position_value,
                "持仓桶数": holding_bucket_count,
            }
        )

    trade_df = pd.DataFrame(trade_records)
    equity_df = pd.DataFrame(equity_records)
    equity_df["净值前高"] = equity_df["净值"].cummax()
    equity_df["回撤"] = equity_df["净值"] / equity_df["净值前高"] - 1
    summary_df = build_summary(f"{bucket_count}桶轮动", trade_df, equity_df, bucket_count)
    return trade_df, equity_df, summary_df


def main() -> None:
    strategy_mod = load_module("bucket_backtest", BACKTEST_MODULE_PATH)
    tail_mod = load_module("bucket_tail_proxy", TAIL_PROXY_PATH)
    base_df, candidate_book = build_best_candidate_book(strategy_mod, tail_mod)
    histories = strategy_mod.fetch_histories(
        sorted(base_df["基础代码"].dropna().astype(str).unique().tolist())
    )

    baseline_trade_df, baseline_equity_df, baseline_summary_df = strategy_mod.run_backtest(candidate_book, histories)
    baseline_row = baseline_summary_df.iloc[0].to_dict()
    baseline_row["方案"] = "当前全仓补位"
    baseline_row["桶数"] = 0
    baseline_row["最大同时持仓桶数"] = strategy_mod.MAX_POSITIONS

    rows = [baseline_row]
    best_bundle = None
    best_return = float("-inf")
    best_ratio = float("-inf")

    for bucket_count in BUCKET_COUNTS:
        trade_df, equity_df, summary_df = run_bucket_backtest(strategy_mod, candidate_book, histories, bucket_count)
        row = summary_df.iloc[0].to_dict()
        rows.append(row)
        total_return = float(row["总收益率"])
        ratio = float(row["收益回撤比"])
        if (total_return > best_return) or (
            math.isclose(total_return, best_return, rel_tol=1e-12, abs_tol=1e-12) and ratio > best_ratio
        ):
            best_return = total_return
            best_ratio = ratio
            best_bundle = (bucket_count, trade_df, equity_df, summary_df)

    result_df = pd.DataFrame(rows).sort_values(["总收益率", "收益回撤比"], ascending=[False, False]).reset_index(drop=True)
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-桶仓实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        bucket_count, trade_df, equity_df, summary_df = best_bundle
        trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
        export_equity_df = equity_df.copy()
        export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
        export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")
        strategy_mod.save_plot(equity_df, Path(f"{output_prefix}-最佳策略净值.png"))
        pd.DataFrame([{"最佳策略": f"{bucket_count}桶轮动"}]).to_csv(
            f"{output_prefix}-最佳策略名称.csv", index=False, encoding="utf-8-sig"
        )

    print("桶仓实验完成")
    print(
        result_df[
            ["方案", "桶数", "期末净值", "总收益率", "最大回撤", "交易笔数", "胜率", "收益回撤比", "最大同时持仓桶数"]
        ].to_string(index=False)
    )
    if best_bundle is not None:
        print(f"\n最佳桶数: {best_bundle[0]}")


if __name__ == "__main__":
    main()
