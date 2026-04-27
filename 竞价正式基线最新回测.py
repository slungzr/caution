from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
OPEN_SECTOR_MODULE_PATH = BASE_DIR / "竞价行业开盘联动探索.py"
PREV_BODY_MIN = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按当前正式基线回测到指定日期")
    parser.add_argument("--end-date", default="20260402", help="截止日期 YYYYMMDD")
    parser.add_argument("--factor-csv", default="", help="可选：指定扩展因子 CSV")
    return parser.parse_args()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_strategy_module():
    return load_module("auction_backtest_latest_baseline", BACKTEST_MODULE_PATH)


def load_open_sector_module():
    return load_module("open_sector_latest_baseline", OPEN_SECTOR_MODULE_PATH)


def prepare_current_base_df(strategy_mod, open_mod, end_date: pd.Timestamp) -> pd.DataFrame:
    sector_mod = open_mod.load_sector_module()
    signal_df, breadth_df = open_mod.prepare_signal_df(strategy_mod, sector_mod)
    signal_df = signal_df[pd.to_datetime(signal_df["日期"]) <= end_date].copy()
    breadth_df = breadth_df[pd.to_datetime(breadth_df["日期"]) <= end_date].copy()

    base_candidate_df = open_mod.build_base_candidate_df(strategy_mod, sector_mod, signal_df, breadth_df)
    base_candidate_df = base_candidate_df[
        pd.to_numeric(base_candidate_df["申万一级行业开盘涨幅"], errors="coerce") > 0
    ].copy()

    prev_body = pd.to_numeric(base_candidate_df["实体涨跌幅前日"], errors="coerce")
    base_candidate_df = base_candidate_df[prev_body >= PREV_BODY_MIN].copy()

    matched_amount = pd.to_numeric(base_candidate_df["竞价匹配金额_openapi"], errors="coerce")
    unmatched_amount = pd.to_numeric(base_candidate_df["竞价未匹配金额"], errors="coerce")
    base_candidate_df["竞价未匹配占比"] = unmatched_amount / matched_amount
    return base_candidate_df.reset_index(drop=True)


def build_candidate_book(strategy_mod, base_df: pd.DataFrame) -> pd.DataFrame:
    sort_by: tuple[tuple[str, bool], ...] = (
        ("竞价未匹配占比", False),
        ("竞昨成交比估算", False),
        (strategy_mod.BUY_RANK_COLUMN, True),
        ("基础代码", True),
    )
    sort_columns = ["日期"]
    ascending = [True]
    for column, is_ascending in sort_by:
        if column not in base_df.columns:
            raise KeyError(f"排序列不存在: {column}")
        sort_columns.append(column)
        ascending.append(is_ascending)

    candidate_book = (
        base_df.sort_values(sort_columns, ascending=ascending, kind="stable")
        .groupby("日期", group_keys=False)
        .head(strategy_mod.MAX_POSITIONS)
        .copy()
    )
    return candidate_book.reset_index(drop=True)


def normalize_daily_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
    }
    df = raw_df.rename(columns=rename_map).copy()
    keep_columns = [column for column in ["date", "open", "high", "low", "close"] if column in df.columns]
    df = df[keep_columns].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
    df["prev_close"] = df["close"].shift(1)
    return df.set_index("date")


def refresh_stock_history_to_end(strategy_mod, code: str, end_date: pd.Timestamp) -> pd.DataFrame:
    symbol = strategy_mod.to_akshare_symbol(code)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            raw_df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date="19900101",
                end_date=end_date.strftime("%Y%m%d"),
                adjust="",
            )
            history_df = normalize_daily_history(raw_df)
            if history_df.empty:
                return history_df
            strategy_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            history_df.reset_index().to_csv(
                strategy_mod.CACHE_DIR / f"{code}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            return history_df
        except Exception as exc:  # pragma: no cover - 依赖网络
            last_error = exc
            time.sleep(attempt)
    print(f"刷新日线失败 {code}: {last_error}")
    return pd.DataFrame()


def ensure_histories_to_end(
    strategy_mod,
    histories: dict[str, pd.DataFrame],
    codes: list[str],
    end_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    refreshed_count = 0
    for code in codes:
        history_df = histories.get(code, pd.DataFrame())
        max_date = pd.Timestamp(history_df.index.max()) if not history_df.empty else pd.NaT
        if pd.isna(max_date) or max_date < end_date:
            refreshed = refresh_stock_history_to_end(strategy_mod, code, end_date)
            if not refreshed.empty:
                histories[code] = refreshed
                refreshed_count += 1
    if refreshed_count:
        print(f"已刷新日线到 {end_date.strftime('%Y%m%d')}: {refreshed_count} 只")
    return histories


def trim_histories(histories: dict[str, pd.DataFrame], end_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    trimmed: dict[str, pd.DataFrame] = {}
    for code, history_df in histories.items():
        if history_df.empty:
            trimmed[code] = history_df
            continue
        clipped_df = history_df[pd.to_datetime(history_df.index) <= end_date].copy()
        trimmed[code] = clipped_df
    return trimmed


def summarize_data_coverage(
    base_df: pd.DataFrame,
    candidate_book: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    history_max_dates = [
        pd.Timestamp(history_df.index.max())
        for history_df in histories.values()
        if not history_df.empty
    ]
    factor_min = pd.to_datetime(base_df["日期"]).min() if not base_df.empty else pd.NaT
    factor_max = pd.to_datetime(base_df["日期"]).max() if not base_df.empty else pd.NaT
    selected_min = pd.to_datetime(candidate_book["日期"]).min() if not candidate_book.empty else pd.NaT
    selected_max = pd.to_datetime(candidate_book["日期"]).max() if not candidate_book.empty else pd.NaT
    return pd.DataFrame(
        [
            {
                "截止日期": end_date.strftime("%Y%m%d"),
                "基线候选宇宙行数": int(len(base_df)),
                "基线候选宇宙交易日": int(base_df["日期"].nunique()) if not base_df.empty else 0,
                "基线候选宇宙起始": factor_min.strftime("%Y%m%d") if pd.notna(factor_min) else "",
                "基线候选宇宙结束": factor_max.strftime("%Y%m%d") if pd.notna(factor_max) else "",
                "入选候选池行数": int(len(candidate_book)),
                "入选候选池交易日": int(candidate_book["日期"].nunique()) if not candidate_book.empty else 0,
                "入选候选池起始": selected_min.strftime("%Y%m%d") if pd.notna(selected_min) else "",
                "入选候选池结束": selected_max.strftime("%Y%m%d") if pd.notna(selected_max) else "",
                "日线股票数": int(len(histories)),
                "日线最早最大日期": min(history_max_dates).strftime("%Y%m%d") if history_max_dates else "",
                "日线最晚最大日期": max(history_max_dates).strftime("%Y%m%d") if history_max_dates else "",
            }
        ]
    )


def export_results(
    strategy_mod,
    candidate_book: pd.DataFrame,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    open_price_check_df: pd.DataFrame,
    end_date: pd.Timestamp,
) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-正式基线至{end_date.strftime('%Y%m%d')}"

    export_candidate_df = candidate_book.copy()
    export_candidate_df["日期"] = pd.to_datetime(export_candidate_df["日期"]).dt.strftime("%Y%m%d")
    keep_columns = [
        "日期",
        "股票代码",
        "股票简称",
        "申万一级行业代码",
        "申万一级行业",
        "申万一级行业开盘涨幅",
        "申万一级行业开盘涨幅排名",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "竞价未匹配占比",
        "竞昨成交比估算",
        "个股热度排名昨日",
        "竞价涨幅今日",
        "开盘价:不复权今日",
        "收盘价:不复权今日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
        "成交量昨日",
        "成交量前日",
        "市场20日高低差",
    ]
    keep_columns = [column for column in keep_columns if column in export_candidate_df.columns]
    export_candidate_df[keep_columns].to_csv(
        f"{output_prefix}-候选池.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trade_df.to_csv(f"{output_prefix}-交易明细.csv", index=False, encoding="utf-8-sig")

    export_equity_df = equity_df.copy()
    export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
    export_equity_df.to_csv(f"{output_prefix}-净值.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(f"{output_prefix}-摘要.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(f"{output_prefix}-数据覆盖.csv", index=False, encoding="utf-8-sig")
    open_price_check_df.to_csv(f"{output_prefix}-开盘价异常.csv", index=False, encoding="utf-8-sig")
    strategy_mod.save_plot(equity_df, Path(f"{output_prefix}-净值.png"))


def main() -> None:
    args = parse_args()
    end_date = pd.Timestamp(args.end_date).normalize()
    strategy_mod = load_strategy_module()
    if args.factor_csv:
        strategy_mod.FACTOR_INPUT_CSV = Path(args.factor_csv)
    open_mod = load_open_sector_module()

    base_df = prepare_current_base_df(strategy_mod, open_mod, end_date)
    if base_df.empty:
        raise RuntimeError("正式基线候选宇宙为空")

    candidate_book = build_candidate_book(strategy_mod, base_df)
    if candidate_book.empty:
        raise RuntimeError("正式基线候选池为空")

    codes = sorted(base_df["基础代码"].dropna().astype(str).unique().tolist())
    histories = strategy_mod.fetch_histories(codes)
    histories = ensure_histories_to_end(strategy_mod, histories, codes, end_date)
    histories = trim_histories(histories, end_date)

    open_price_check_df = strategy_mod.build_open_price_check(candidate_book, histories)
    trade_df, equity_df, summary_df = strategy_mod.run_backtest(candidate_book, histories)
    coverage_df = summarize_data_coverage(base_df, candidate_book, histories, end_date)
    export_results(
        strategy_mod,
        candidate_book,
        trade_df,
        equity_df,
        summary_df,
        coverage_df,
        open_price_check_df,
        end_date,
    )

    print("正式基线最新回测完成")
    print("数据覆盖：")
    print(coverage_df.to_string(index=False))
    print("\n摘要：")
    print(summary_df.to_string(index=False))
    print(f"\n开盘价异常数: {len(open_price_check_df)}")
    if not open_price_check_df.empty:
        print(open_price_check_df.head(20).to_string(index=False))
    print("\n最后10条候选：")
    display_columns = [
        "日期",
        "股票代码",
        "股票简称",
        "申万一级行业",
        "申万一级行业开盘涨幅",
        "竞价未匹配占比",
        "竞昨成交比估算",
        "个股热度排名昨日",
    ]
    display_columns = [column for column in display_columns if column in candidate_book.columns]
    display_df = candidate_book[display_columns].tail(10).copy()
    display_df["日期"] = pd.to_datetime(display_df["日期"]).dt.strftime("%Y%m%d")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
