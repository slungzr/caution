#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Explore how yesterday's float market cap affects the auction backtest."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_SCRIPT = BASE_DIR / "竞价爬升策略回测.py"
PLAYWRIGHT_SCRIPT = BASE_DIR / "问财_playwright抓取.py"
TRADE_CALENDAR_CSV = BASE_DIR / "trade_calendar.csv"
CACHE_DIR = BASE_DIR / "cache" / "wencai_market_cap"
OUTPUT_PREFIX = BASE_DIR / "竞价爬升-20240504-市值影响探索"
MARKET_CAP_COLUMN = "昨日a股流通市值"


@dataclass(frozen=True)
class MarketCapBin:
    name: str
    filters: dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探索昨日流通市值对竞价策略回测收益的影响")
    parser.add_argument("--factor-csv", default="", help="扩展因子 CSV，默认优先使用带市值的至20260423文件")
    parser.add_argument(
        "--config",
        choices=["auction-ratio", "new-factor"],
        default="auction-ratio",
        help="使用原回测里的哪套策略配置",
    )
    parser.add_argument("--fetch-missing-cap", action="store_true", default=True, help="缺少市值字段时用 Playwright 补齐")
    parser.add_argument("--no-fetch-missing-cap", action="store_false", dest="fetch_missing_cap", help="缺少市值字段时直接报错")
    parser.add_argument("--profile-dir", default=".playwright_wencai_profile", help="Playwright Chrome 用户目录")
    parser.add_argument("--timeout", type=int, default=90, help="Playwright 单次问财查询超时秒数")
    parser.add_argument("--headless", action="store_true", help="Playwright 无界面运行")
    parser.add_argument("--max-pages", type=int, default=3, help="Playwright 分页上限")
    parser.add_argument("--expanded-perpage", type=int, default=100, help="Playwright 尝试单页行数")
    parser.add_argument(
        "--scan-mode",
        choices=["fixed", "threshold", "both"],
        default="both",
        help="市值实验类型：固定分档、下限阈值扫描，或两者都跑",
    )
    parser.add_argument(
        "--fill-existing-cap",
        action="store_true",
        help="已有市值列但存在空值时，也用 Playwright 按日期补齐缺口",
    )
    parser.add_argument("--cap-start-date", default="", help="补市值起始交易日 YYYYMMDD，仅限制 Playwright 补缺")
    parser.add_argument("--cap-end-date", default="", help="补市值结束交易日 YYYYMMDD，仅限制 Playwright 补缺")
    parser.add_argument("--max-cap-fetch-dates", type=int, default=0, help="本次最多补多少个交易日，0 表示不限制")
    parser.add_argument("--cap-only", action="store_true", help="只补齐/缓存市值，不执行回测")
    return parser.parse_args()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_matplotlib_stub_if_needed() -> None:
    try:
        import matplotlib.pyplot  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.rcParams = {}
    pyplot.subplots = lambda *args, **kwargs: (_raise_missing_matplotlib())
    pyplot.close = lambda *args, **kwargs: None
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)


def _raise_missing_matplotlib():
    raise ModuleNotFoundError("matplotlib is required only when plotting backtest equity curves")


def load_backtest_module():
    install_matplotlib_stub_if_needed()
    return load_module("auction_backtest_market_cap", BACKTEST_SCRIPT)


def load_playwright_module():
    return load_module("wencai_playwright_market_cap", PLAYWRIGHT_SCRIPT)


def choose_factor_csv(args: argparse.Namespace) -> Path:
    if args.factor_csv:
        path = Path(args.factor_csv)
        return path if path.is_absolute() else BASE_DIR / path
    candidates = [
        BASE_DIR / "竞价爬升-20240504-扩展因子-至20260423.csv",
        BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到扩展因子 CSV")


def normalize_amount_series(values: pd.Series) -> pd.Series:
    def convert(value: Any) -> float:
        if value is None or pd.isna(value):
            return math.nan
        text = str(value).strip().replace(",", "")
        if not text or text in {"-", "--", "nan", "None"}:
            return math.nan
        multiplier = 1.0
        if text.endswith("万"):
            multiplier = 10_000.0
            text = text[:-1]
        elif text.endswith("亿"):
            multiplier = 100_000_000.0
            text = text[:-1]
        text = text.replace("元", "").strip()
        try:
            return float(text) * multiplier
        except ValueError:
            return math.nan

    return values.map(convert)


def find_market_cap_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "a股市值(不含限售股)昨日",
        "昨日a股流通市值",
        "a股流通市值昨日",
        "流通市值昨日",
        "昨日流通市值",
        "总市值昨日",
    ]
    for column in candidates:
        if column in df.columns:
            return column

    market_cap_columns = [
        column
        for column in df.columns
        if "市值" in str(column) and ("昨日" in str(column) or "流通" in str(column) or "不含限售" in str(column))
    ]
    if market_cap_columns:
        market_cap_columns.sort(key=lambda name: (0 if "流通" in str(name) or "不含限售" in str(name) else 1, len(str(name))))
        return market_cap_columns[0]
    return None


def build_fixed_market_cap_bins(column: str = MARKET_CAP_COLUMN) -> list[MarketCapBin]:
    yi = 100_000_000.0
    return [
        MarketCapBin("全样本", {}),
        MarketCapBin("<20亿", {column: {"max": 20 * yi}}),
        MarketCapBin("20-50亿", {column: {"min": 20 * yi, "max": 50 * yi}}),
        MarketCapBin("50-100亿", {column: {"min": 50 * yi, "max": 100 * yi}}),
        MarketCapBin("100-200亿", {column: {"min": 100 * yi, "max": 200 * yi}}),
        MarketCapBin(">=200亿", {column: {"min": 200 * yi}}),
    ]


def build_threshold_market_cap_bins(
    column: str = MARKET_CAP_COLUMN,
    thresholds_yi: list[int] | None = None,
) -> list[MarketCapBin]:
    yi = 100_000_000.0
    thresholds_yi = thresholds_yi or [20, 30, 50, 80, 100, 150, 200, 300]
    bins = [MarketCapBin("全样本", {})]
    bins.extend(MarketCapBin(f">={threshold}亿", {column: {"min": threshold * yi}}) for threshold in thresholds_yi)
    return bins


def build_market_cap_bins(scan_mode: str, column: str = MARKET_CAP_COLUMN) -> list[MarketCapBin]:
    bins: list[MarketCapBin] = []
    if scan_mode in {"fixed", "both"}:
        bins.extend(MarketCapBin(f"固定分档:{item.name}", item.filters) for item in build_fixed_market_cap_bins(column))
    if scan_mode in {"threshold", "both"}:
        bins.extend(MarketCapBin(f"阈值扫描:{item.name}", item.filters) for item in build_threshold_market_cap_bins(column))

    seen: set[str] = set()
    unique_bins: list[MarketCapBin] = []
    for item in bins:
        key = f"{item.name}|{item.filters}"
        if key in seen:
            continue
        seen.add(key)
        unique_bins.append(item)
    return unique_bins


def parse_optional_date(value: str) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(value).normalize()


def rows_needing_market_cap_fetch(
    df: pd.DataFrame,
    column: str = MARKET_CAP_COLUMN,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if column not in df.columns:
        needs_fetch = df.copy()
    else:
        needs_fetch = df[df[column].isna()].copy()
    if needs_fetch.empty:
        return needs_fetch

    dates = pd.to_datetime(needs_fetch["日期"], errors="coerce").dt.normalize()
    mask = dates.notna()
    if start_date is not None:
        mask &= dates >= pd.Timestamp(start_date).normalize()
    if end_date is not None:
        mask &= dates <= pd.Timestamp(end_date).normalize()
    return needs_fetch[mask].copy()


def load_trade_calendar() -> list[pd.Timestamp]:
    if not TRADE_CALENDAR_CSV.exists():
        return []
    df = pd.read_csv(TRADE_CALENDAR_CSV, encoding="utf-8-sig")
    date_column = "日期" if "日期" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[date_column].astype(str), errors="coerce").dropna()
    return sorted(pd.Timestamp(date).normalize() for date in dates.unique())


def previous_trade_date(trade_date: pd.Timestamp, calendar: list[pd.Timestamp]) -> pd.Timestamp:
    trade_date = pd.Timestamp(trade_date).normalize()
    prior = [date for date in calendar if date < trade_date]
    if prior:
        return prior[-1]
    return (trade_date - pd.tseries.offsets.BDay(1)).normalize()


def cn_date(date: pd.Timestamp) -> str:
    date = pd.Timestamp(date)
    return f"{date.year}年{date.month}月{date.day}日"


def market_cap_query(prev_trade_date: pd.Timestamp) -> str:
    date_text = cn_date(prev_trade_date)
    return f"{date_text}a股流通市值，{date_text}个股热度排名前100"


def code_market_cap_query(codes: list[str], prev_trade_date: pd.Timestamp) -> str:
    date_text = cn_date(prev_trade_date)
    code_text = "、".join(str(code) for code in codes)
    return f"{code_text}，{date_text}a股流通市值"


def chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def standardize_market_cap_frame(df: pd.DataFrame, trade_date: pd.Timestamp, normalize_code) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["日期", "基础代码", MARKET_CAP_COLUMN])
    columns = list(df.columns)
    code_column = next(
        (
            column
            for column in columns
            if "股票代码" in str(column)
            or str(column).strip().lower() == "code"
            or str(column).strip() == "代码"
            or str(column).strip() == "证券代码"
        ),
        None,
    )
    cap_column = find_market_cap_column(df)
    if code_column is None or cap_column is None:
        raise KeyError(f"问财市值结果缺少股票代码或市值列: {columns}")

    out = pd.DataFrame(
        {
            "日期": pd.Timestamp(trade_date).normalize(),
            "基础代码": df[code_column].map(normalize_code),
            MARKET_CAP_COLUMN: normalize_amount_series(df[cap_column]),
        }
    )
    out = out.dropna(subset=["基础代码", MARKET_CAP_COLUMN])
    out = out[out["基础代码"].astype(str) != ""].drop_duplicates(subset=["日期", "基础代码"], keep="last")
    return out.reset_index(drop=True)


def load_cached_market_cap(trade_date: pd.Timestamp) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{pd.Timestamp(trade_date):%Y%m%d}.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(cache_path, encoding="utf-8-sig")
    if df.empty:
        return df
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.normalize()
    df[MARKET_CAP_COLUMN] = normalize_amount_series(df[MARKET_CAP_COLUMN])
    return df.dropna(subset=["日期", "基础代码", MARKET_CAP_COLUMN]).reset_index(drop=True)


def save_cached_market_cap(trade_date: pd.Timestamp, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{pd.Timestamp(trade_date):%Y%m%d}.csv"
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")


def fetch_market_cap_with_playwright(
    signal_df: pd.DataFrame,
    args: argparse.Namespace,
    normalize_code,
) -> pd.DataFrame:
    calendar = load_trade_calendar()
    dates = sorted(pd.Timestamp(date).normalize() for date in signal_df["日期"].dropna().unique())
    frames: list[pd.DataFrame] = []
    work_items: list[tuple[pd.Timestamp, pd.DataFrame, list[str]]] = []
    for trade_date in dates:
        cached = load_cached_market_cap(trade_date)
        date_signal = signal_df[pd.to_datetime(signal_df["日期"], errors="coerce").dt.normalize() == trade_date]
        required_codes = sorted(date_signal["基础代码"].dropna().astype(str).unique().tolist())
        cached_codes = set(cached["基础代码"].dropna().astype(str).tolist()) if not cached.empty else set()
        missing_codes = [code for code in required_codes if code not in cached_codes]
        if cached.empty or missing_codes:
            work_items.append((trade_date, cached, missing_codes))
        else:
            frames.append(cached)

    if args.max_cap_fetch_dates and args.max_cap_fetch_dates > 0:
        skipped_items = work_items[args.max_cap_fetch_dates :]
        for _trade_date, cached, _missing_codes in skipped_items:
            if not cached.empty:
                frames.append(cached)
        work_items = work_items[: args.max_cap_fetch_dates]

    if not work_items:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    playwright_mod = load_playwright_module()
    with playwright_mod.WencaiPlaywrightClient(
        args.profile_dir,
        timeout=args.timeout,
        headless=args.headless,
        max_pages=args.max_pages,
        expanded_perpage=args.expanded_perpage,
    ) as client:
        for trade_date, cached, missing_codes in work_items:
            prev_date = previous_trade_date(trade_date, calendar)
            pieces: list[pd.DataFrame] = []
            if not cached.empty:
                pieces.append(cached)

            if cached.empty:
                query = market_cap_query(prev_date)
                df, _payload, url = client.query(query)
                cap_df = standardize_market_cap_frame(df, trade_date, normalize_code)
                pieces.append(cap_df)
                print(f"Playwright市值完成 date={trade_date:%Y%m%d} rows={len(cap_df)} url={url}")
                time.sleep(1.2)

            combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            cached_codes = set(combined["基础代码"].dropna().astype(str).tolist()) if not combined.empty else set()
            date_signal = signal_df[pd.to_datetime(signal_df["日期"], errors="coerce").dt.normalize() == trade_date]
            required_codes = sorted(date_signal["基础代码"].dropna().astype(str).unique().tolist())
            missing_codes = [code for code in required_codes if code not in cached_codes]

            for code_chunk in chunk_list(missing_codes, 30):
                query = code_market_cap_query(code_chunk, prev_date)
                df, _payload, url = client.query(query)
                cap_df = standardize_market_cap_frame(df, trade_date, normalize_code)
                pieces.append(cap_df)
                print(
                    f"Playwright代码市值完成 date={trade_date:%Y%m%d} "
                    f"codes={len(code_chunk)} rows={len(cap_df)} url={url}"
                )
                time.sleep(1.2)

            cap_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            if not cap_df.empty:
                cap_df = cap_df.drop_duplicates(subset=["日期", "基础代码"], keep="last").reset_index(drop=True)
            save_cached_market_cap(trade_date, cap_df)
            frames.append(cap_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_market_cap(signal_df: pd.DataFrame, args: argparse.Namespace, normalize_code) -> pd.DataFrame:
    df = signal_df.copy()
    existing_column = find_market_cap_column(df)
    if existing_column is not None:
        df[MARKET_CAP_COLUMN] = normalize_amount_series(df[existing_column])
        missing_df = rows_needing_market_cap_fetch(
            df,
            MARKET_CAP_COLUMN,
            parse_optional_date(args.cap_start_date),
            parse_optional_date(args.cap_end_date),
        )
        if not missing_df.empty and args.fill_existing_cap:
            cap_df = fetch_market_cap_with_playwright(missing_df, args, normalize_code)
            if not cap_df.empty:
                df = df.merge(
                    cap_df.rename(columns={MARKET_CAP_COLUMN: "_playwright_market_cap"}),
                    on=["日期", "基础代码"],
                    how="left",
                )
                df[MARKET_CAP_COLUMN] = df[MARKET_CAP_COLUMN].fillna(df["_playwright_market_cap"])
                df = df.drop(columns=["_playwright_market_cap"])
        elif not missing_df.empty:
            print(
                f"市值字段已有，但仍有 {len(missing_df)} 行缺失；"
                "如需补齐历史缺口，运行时加 --fill-existing-cap"
            )
        return df

    if not args.fetch_missing_cap:
        raise KeyError("扩展因子文件缺少市值字段，且未启用 Playwright 补齐")

    cap_df = fetch_market_cap_with_playwright(df, args, normalize_code)
    if cap_df.empty:
        raise RuntimeError("Playwright 未能补齐市值数据")

    merged = df.merge(cap_df, on=["日期", "基础代码"], how="left")
    missing = int(merged[MARKET_CAP_COLUMN].isna().sum())
    if missing:
        print(f"市值补齐后仍缺失 {missing} 行，将在分档过滤时自动剔除")
    return merged


def merge_filter_rules(*rules: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rule in rules:
        if not rule:
            continue
        for column, config in rule.items():
            merged[column] = dict(config)
    return merged


def select_strategy_config(backtest_mod, name: str):
    if name == "new-factor":
        return backtest_mod.NEW_FACTOR_CONFIG
    return backtest_mod.AUCTION_RATIO_RANK_CONFIG


def prepare_market_cap_configs(backtest_mod, base_config, bins: list[MarketCapBin]) -> list[Any]:
    configs = []
    for cap_bin in bins:
        filters = merge_filter_rules(base_config.buy_filters, cap_bin.filters)
        configs.append(
            backtest_mod.StrategyConfig(
                name=f"{base_config.name}-{cap_bin.name}",
                buy_filters=filters,
                named_conditions=base_config.named_conditions,
                market_filter=base_config.market_filter,
                sort_by=base_config.sort_by,
            )
        )
    return configs


def build_candidate_books(backtest_mod, signal_df: pd.DataFrame, breadth_df: pd.DataFrame, configs: list[Any]) -> dict[str, pd.DataFrame]:
    books: dict[str, pd.DataFrame] = {}
    for config in configs:
        try:
            candidate_book = backtest_mod.build_candidate_book_with_config(
                signal_df=signal_df,
                breadth_df=breadth_df,
                buy_filters=config.buy_filters,
                named_conditions=config.named_conditions,
                market_filter=config.market_filter,
                sort_by=config.sort_by,
                max_positions=backtest_mod.MAX_POSITIONS,
            )
        except Exception as exc:
            print(f"候选池构建失败 {config.name}: {exc}")
            candidate_book = pd.DataFrame()
        books[config.name] = candidate_book
    return books


def run_experiment() -> pd.DataFrame:
    args = parse_args()
    backtest_mod = load_backtest_module()
    factor_csv = choose_factor_csv(args)
    signal_df = backtest_mod.load_factor_signal_data(factor_csv)
    signal_df = attach_market_cap(signal_df, args, backtest_mod.normalize_stock_code)
    enriched_path = BASE_DIR / f"{OUTPUT_PREFIX.name}-市值补齐样本.csv"
    signal_df.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    if args.cap_only:
        missing_count = int(signal_df[MARKET_CAP_COLUMN].isna().sum()) if MARKET_CAP_COLUMN in signal_df.columns else len(signal_df)
        print(f"市值补齐样本: {enriched_path}")
        print(f"剩余市值缺失行数: {missing_count}")
        return pd.DataFrame()

    signal_df = signal_df[signal_df[MARKET_CAP_COLUMN].notna()].copy()
    if signal_df.empty:
        raise ValueError("市值字段全部为空，无法探索")

    breadth_df = backtest_mod.fetch_market_breadth_history(backtest_mod.MARKET_BREADTH_SYMBOL)
    base_config = select_strategy_config(backtest_mod, args.config)
    bins = build_market_cap_bins(args.scan_mode, MARKET_CAP_COLUMN)
    configs = prepare_market_cap_configs(backtest_mod, base_config, bins)
    bin_name_by_config = {config.name: cap_bin.name for config, cap_bin in zip(configs, bins)}
    candidate_books = build_candidate_books(backtest_mod, signal_df, breadth_df, configs)

    all_codes = sorted(
        {
            str(code)
            for book in candidate_books.values()
            if not book.empty
            for code in book["基础代码"].dropna().astype(str).tolist()
        }
    )
    histories = backtest_mod.fetch_histories(all_codes)

    rows: list[dict[str, Any]] = []
    best_name = ""
    best_score = -math.inf
    best_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    for config in configs:
        candidate_book = candidate_books[config.name]
        if candidate_book.empty:
            rows.append(
                {
                    "策略名": config.name,
                    "市值区间": bin_name_by_config.get(config.name, ""),
                    "错误": "候选池为空",
                    "候选池行数": 0,
                    "候选交易日": 0,
                }
            )
            continue

        trade_df, equity_df, summary_df = backtest_mod.run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        market_cap = pd.to_numeric(candidate_book[MARKET_CAP_COLUMN], errors="coerce")
        row.update(
            {
                "策略名": config.name,
                "市值字段": MARKET_CAP_COLUMN,
                "市值区间": bin_name_by_config.get(config.name, ""),
                "因子文件": str(factor_csv),
                "候选池行数": int(len(candidate_book)),
                "候选交易日": int(candidate_book["日期"].nunique()),
                "候选市值中位数": float(market_cap.median()),
                "候选市值最小值": float(market_cap.min()),
                "候选市值最大值": float(market_cap.max()),
            }
        )
        rows.append(row)
        score = row.get("收益回撤比")
        if pd.notna(score) and float(score) > best_score:
            best_score = float(score)
            best_name = config.name
            best_outputs = (candidate_book, trade_df, equity_df)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["收益回撤比", "总收益率"], ascending=[False, False], na_position="last")
    result_df.to_csv(f"{OUTPUT_PREFIX}-结果.csv", index=False, encoding="utf-8-sig")

    if best_outputs is not None:
        candidate_book, trade_df, equity_df = best_outputs
        candidate_export = candidate_book.copy()
        candidate_export["日期"] = pd.to_datetime(candidate_export["日期"]).dt.strftime("%Y%m%d")
        candidate_export.to_csv(f"{OUTPUT_PREFIX}-最佳策略候选池.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(f"{OUTPUT_PREFIX}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
        equity_export = equity_df.copy()
        equity_export["日期"] = pd.to_datetime(equity_export["日期"]).dt.strftime("%Y%m%d")
        equity_export.to_csv(f"{OUTPUT_PREFIX}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
        (BASE_DIR / f"{OUTPUT_PREFIX.name}-最佳策略名称.txt").write_text(best_name, encoding="utf-8")

    print(f"因子文件: {factor_csv}")
    print(f"结果: {OUTPUT_PREFIX}-结果.csv")
    if best_name:
        print(f"最佳策略: {best_name}")
    print(result_df.head(10).to_string(index=False))
    return result_df


if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        raise SystemExit(130)
