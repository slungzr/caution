from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from wencai_openapi import WencaiOpenAPIError, query_split_and_merge


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "竞价爬升-20240504.csv"
OUTPUT_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子.csv"
FAILED_OUTPUT_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-失败日期.csv"
CACHE_DIR = BASE_DIR / "cache" / "wencai_openapi_factors"
DEFAULT_QUERY_LIMIT = 100

FACTOR_COLUMN_MAP = {
    "股票代码": "股票代码",
    "股票简称": "股票简称",
    "竞价强度": "竞价强度",
    "竞价匹配金额": "竞价匹配金额_openapi",
    "竞价未匹配金额": "竞价未匹配金额",
    "竞价换手率": "竞价换手率_openapi",
    "竞价量比": "竞价量比_openapi",
    "涨停开板": "昨日炸板_openapi",
    "涨停": "昨日涨停_openapi",
    "连续涨停天数": "连续涨停天数_openapi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补充竞价相关因子")
    parser.add_argument("--input", default=str(INPUT_CSV), help="输入 CSV 路径")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="输出 CSV 路径")
    parser.add_argument("--failed-output", default=str(FAILED_OUTPUT_CSV), help="失败日期输出路径")
    parser.add_argument("--start-date", default="", help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", default="", help="结束日期 YYYYMMDD")
    parser.add_argument("--date-limit", type=int, default=0, help="仅处理前 N 个交易日")
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="问财分页大小")
    parser.add_argument("--sleep", type=float, default=0.3, help="每个交易日查询后的暂停秒数")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略本地缓存重新拉取")
    return parser.parse_args()


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["gb18030", "gbk", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"读取 CSV 失败: {csv_path}")


def normalize_stock_code(value: Any) -> str:
    digits = re.findall(r"\d+", str(value))
    return "".join(digits)[-6:].zfill(6) if digits else ""


def date_to_cn_text(date_value: pd.Timestamp) -> str:
    return f"{date_value.year}年{date_value.month}月{date_value.day}日"


def build_queries(trade_date: pd.Timestamp, prev_trade_date: pd.Timestamp) -> list[str]:
    today_text = date_to_cn_text(trade_date)
    yesterday_text = date_to_cn_text(prev_trade_date)
    return [
        f"{today_text}竞价强度，{today_text}竞价金额，{today_text}竞价未匹配金额，{yesterday_text}个股热度排名前100",
        f"{today_text}竞价换手率，{today_text}竞价量比，{yesterday_text}个股热度排名前100",
        f"{yesterday_text}涨停，{yesterday_text}个股热度排名前100",
        f"{yesterday_text}曾涨停，{yesterday_text}个股热度排名前100",
        f"{yesterday_text}连续涨停天数，{yesterday_text}个股热度排名前100",
    ]


def pick_column(columns: list[str], keyword: str) -> str | None:
    exact_matches = [column for column in columns if column == keyword]
    if exact_matches:
        return exact_matches[0]
    matches = [column for column in columns if keyword in column]
    if not matches:
        return None
    matches.sort(key=len)
    return matches[0]


def standardize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    columns = list(df.columns)
    for keyword, target_name in FACTOR_COLUMN_MAP.items():
        source_column = pick_column(columns, keyword)
        if source_column is not None:
            rename_map[source_column] = target_name

    normalized = df.rename(columns=rename_map).copy()
    if "股票代码" not in normalized.columns:
        raise RuntimeError("问财结果缺少股票代码列")
    if "股票简称" not in normalized.columns:
        normalized["股票简称"] = ""
    normalized["基础代码"] = normalized["股票代码"].apply(normalize_stock_code)
    keep_columns = [
        column
        for column in [
            "股票代码",
            "股票简称",
            "基础代码",
            "竞价强度",
            "竞价匹配金额_openapi",
            "竞价未匹配金额",
            "竞价换手率_openapi",
            "竞价量比_openapi",
            "昨日涨停_openapi",
            "昨日炸板_openapi",
            "连续涨停天数_openapi",
        ]
        if column in normalized.columns
    ]
    return normalized[keep_columns].copy()


def load_cached_factors(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_csv(cache_path, encoding="utf-8-sig")


def fetch_factors_for_date(
    trade_date: pd.Timestamp,
    prev_trade_date: pd.Timestamp,
    limit: int,
    refresh_cache: bool,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{trade_date.strftime('%Y%m%d')}.csv"
    if not refresh_cache and cache_path.exists():
        cached_df = load_cached_factors(cache_path)
        if not cached_df.empty:
            return cached_df

    merged_df = query_split_and_merge(build_queries(trade_date, prev_trade_date), limit=limit, how="outer")
    standardized_df = standardize_factor_columns(merged_df)
    standardized_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return standardized_df


def build_date_list(df: pd.DataFrame, start_date: str, end_date: str, date_limit: int) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce").dropna().unique().tolist())
    if start_date:
        dates = [date for date in dates if date >= pd.Timestamp(start_date)]
    if end_date:
        dates = [date for date in dates if date <= pd.Timestamp(end_date)]
    if date_limit > 0:
        dates = dates[:date_limit]
    return [pd.Timestamp(date) for date in dates]


def build_prev_trade_date_map(df: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    all_dates = sorted(pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce").dropna().unique().tolist())
    date_map: dict[pd.Timestamp, pd.Timestamp] = {}
    previous_date: pd.Timestamp | None = None
    for date_value in all_dates:
        current = pd.Timestamp(date_value)
        if previous_date is not None:
            date_map[current] = previous_date
        previous_date = current
    return date_map


def enrich_dataframe(
    source_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    date_limit: int,
    limit: int,
    sleep_seconds: float,
    refresh_cache: bool,
) -> pd.DataFrame:
    working_df = source_df.copy()
    working_df["日期"] = working_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    working_df["基础代码"] = working_df["股票代码"].apply(normalize_stock_code)
    working_df["日期_dt"] = pd.to_datetime(working_df["日期"].astype(str), format="%Y%m%d", errors="coerce")

    target_dates = build_date_list(working_df, start_date, end_date, date_limit)
    prev_trade_date_map = build_prev_trade_date_map(working_df)
    factor_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for index, trade_date in enumerate(target_dates, start=1):
        print(f"[{index}/{len(target_dates)}] 拉取 {trade_date.strftime('%Y%m%d')} 因子")
        prev_trade_date = prev_trade_date_map.get(trade_date)
        if prev_trade_date is None:
            failures.append({"日期": trade_date.strftime("%Y%m%d"), "错误": "缺少上一交易日，跳过"})
            continue
        try:
            factor_df = fetch_factors_for_date(
                trade_date=trade_date,
                prev_trade_date=prev_trade_date,
                limit=limit,
                refresh_cache=refresh_cache,
            )
            factor_df["日期"] = trade_date.strftime("%Y%m%d")
            factor_frames.append(factor_df)
        except WencaiOpenAPIError as exc:
            failures.append({"日期": trade_date.strftime("%Y%m%d"), "错误": str(exc)})
        except Exception as exc:
            failures.append({"日期": trade_date.strftime("%Y%m%d"), "错误": str(exc)})
        time.sleep(max(0.0, sleep_seconds))

    if factor_frames:
        factor_all_df = pd.concat(factor_frames, ignore_index=True)
        merged_df = working_df.merge(
            factor_all_df,
            on=["日期", "基础代码"],
            how="left",
            suffixes=("", "_factor"),
        )
    else:
        merged_df = working_df.copy()

    if "股票代码_factor" in merged_df.columns:
        merged_df.drop(columns=["股票代码_factor"], inplace=True)
    if "股票简称_factor" in merged_df.columns:
        merged_df.drop(columns=["股票简称_factor"], inplace=True)

    merged_df.drop(columns=["日期_dt"], inplace=True)
    if failures:
        print("以下交易日拉取失败:")
        for item in failures:
            print(f"{item['日期']}: {item['错误']}")
    return merged_df, pd.DataFrame(failures)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_output_path = Path(args.failed_output)
    source_df = read_csv_with_fallback(input_path)
    enriched_df, failed_df = enrich_dataframe(
        source_df=source_df,
        start_date=args.start_date,
        end_date=args.end_date,
        date_limit=args.date_limit,
        limit=args.limit,
        sleep_seconds=args.sleep,
        refresh_cache=args.refresh_cache,
    )
    enriched_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_output_path, index=False, encoding="utf-8-sig")
    print(f"输出完成: {output_path}")
    print(f"失败日期清单: {failed_output_path}")
    print(f"行数: {len(enriched_df)}")


if __name__ == "__main__":
    main()
