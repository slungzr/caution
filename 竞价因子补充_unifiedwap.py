from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from wencai_unifiedwap import WencaiUnifiedWapError, merge_frames, query_unified_wap


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "竞价爬升-20240504.csv"
EXISTING_FACTOR_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
FAILED_DATES_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-失败日期.csv"
OUTPUT_CSV = EXISTING_FACTOR_CSV
CACHE_DIR = BASE_DIR / "cache" / "wencai_unifiedwap_factors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 unified-wap 低频补充竞价因子")
    parser.add_argument("--input", default=str(INPUT_CSV))
    parser.add_argument("--existing", default=str(EXISTING_FACTOR_CSV))
    parser.add_argument("--failed-dates", default=str(FAILED_DATES_CSV))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--date-limit", type=int, default=0)
    parser.add_argument("--sleep-query", type=float, default=3.0)
    parser.add_argument("--sleep-date", type=float, default=4.0)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"读取失败: {csv_path}")


def normalize_stock_code(value: Any) -> str:
    digits = re.findall(r"\d+", str(value))
    return "".join(digits)[-6:].zfill(6) if digits else ""


def date_to_cn_text(date_value: pd.Timestamp) -> str:
    return f"{date_value.year}年{date_value.month}月{date_value.day}日"


def build_prev_trade_date_map(df: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    all_dates = sorted(pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce").dropna().unique().tolist())
    result: dict[pd.Timestamp, pd.Timestamp] = {}
    prev = None
    for current in all_dates:
        current_ts = pd.Timestamp(current)
        if prev is not None:
            result[current_ts] = prev
        prev = current_ts
    return result


def pick_column(columns: list[str], keyword: str) -> str | None:
    exact = [column for column in columns if column == keyword]
    if exact:
        return exact[0]
    matches = [column for column in columns if keyword in column]
    if not matches:
        return None
    matches.sort(key=len)
    return matches[0]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    rename_map: dict[str, str] = {}
    for keyword, target in [
        ("股票代码", "股票代码"),
        ("股票简称", "股票简称"),
        ("个股热度排名", "个股热度排名_openapi"),
        ("竞价强度", "竞价强度"),
        ("竞价金额", "竞价匹配金额_openapi"),
        ("竞价未匹配金额", "竞价未匹配金额"),
        ("分时换手率", "竞价换手率_openapi"),
        ("集合竞价评级", "集合竞价评级_unifiedwap"),
        ("竞价强度排名", "竞价强度排名_unifiedwap"),
    ]:
        source = pick_column(columns, keyword)
        if source is not None:
            rename_map[source] = target

    normalized = df.rename(columns=rename_map).copy()
    code_column = "股票代码" if "股票代码" in normalized.columns else "code"
    if code_column != "股票代码":
        normalized.rename(columns={code_column: "股票代码"}, inplace=True)
    if "股票简称" not in normalized.columns:
        normalized["股票简称"] = ""
    normalized["基础代码"] = normalized["股票代码"].apply(normalize_stock_code)

    keep = [
        column
        for column in [
            "股票代码",
            "股票简称",
            "基础代码",
            "个股热度排名_openapi",
            "竞价强度",
            "竞价匹配金额_openapi",
            "竞价未匹配金额",
            "竞价换手率_openapi",
            "集合竞价评级_unifiedwap",
            "竞价强度排名_unifiedwap",
        ]
        if column in normalized.columns
    ]
    return normalized[keep].copy()


def build_questions(trade_date: pd.Timestamp, prev_trade_date: pd.Timestamp) -> list[str]:
    today = date_to_cn_text(trade_date)
    prev = date_to_cn_text(prev_trade_date)
    return [
        f"{today}竞价强度 *1，{prev}个股热度排名前100",
        f"{today}竞价金额，{prev}个股热度排名前100",
        f"{today}竞价换手率，{prev}个股热度排名前100",
    ]


def fetch_date_factors(trade_date: pd.Timestamp, prev_trade_date: pd.Timestamp, refresh_cache: bool, sleep_query: float) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{trade_date.strftime('%Y%m%d')}.csv"
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        if not cached.empty:
            return cached

    frames: list[pd.DataFrame] = []
    for question in build_questions(trade_date, prev_trade_date):
        frame = query_unified_wap(question, perpage=100)
        frames.append(frame)
        time.sleep(max(0.0, sleep_query))

    merged = merge_frames(frames, how="outer")
    standardized = standardize_columns(merged)
    standardized.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return standardized


def main() -> None:
    args = parse_args()
    source_df = read_csv_with_fallback(Path(args.input))
    source_df["日期"] = source_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    source_df["基础代码"] = source_df["股票代码"].apply(normalize_stock_code)

    existing_df = read_csv_with_fallback(Path(args.existing))
    existing_df["日期"] = existing_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    existing_df["基础代码"] = existing_df["基础代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

    failed_df = read_csv_with_fallback(Path(args.failed_dates))
    all_failed_dates = [pd.Timestamp(value) for value in failed_df["日期"].astype(str).tolist() if str(value).strip()]
    target_dates = all_failed_dates.copy()
    if args.date_limit > 0:
        target_dates = target_dates[: args.date_limit]
    untouched_dates = {date.strftime("%Y%m%d") for date in all_failed_dates[len(target_dates):]}

    prev_trade_date_map = build_prev_trade_date_map(source_df)
    new_factor_frames: list[pd.DataFrame] = []
    succeeded_dates: list[str] = []
    failed_rows: list[dict[str, str]] = []

    for index, trade_date in enumerate(target_dates, start=1):
        date_text = trade_date.strftime("%Y%m%d")
        print(f"[{index}/{len(target_dates)}] 补 {date_text}")
        prev_trade_date = prev_trade_date_map.get(trade_date)
        if prev_trade_date is None:
            failed_rows.append({"日期": date_text, "错误": "缺少上一交易日"})
            continue
        try:
            factor_df = fetch_date_factors(trade_date, prev_trade_date, args.refresh_cache, args.sleep_query)
            factor_df["日期"] = date_text
            new_factor_frames.append(factor_df)
            succeeded_dates.append(date_text)
        except WencaiUnifiedWapError as exc:
            failed_rows.append({"日期": date_text, "错误": str(exc)})
        except Exception as exc:
            failed_rows.append({"日期": date_text, "错误": str(exc)})
        time.sleep(max(0.0, args.sleep_date))

    if new_factor_frames:
        factor_all = pd.concat(new_factor_frames, ignore_index=True)
        updated_df = existing_df.merge(factor_all, on=["日期", "基础代码"], how="left", suffixes=("", "__new"))
        for column in factor_all.columns:
            if column in {"日期", "基础代码"}:
                continue
            new_column = f"{column}__new"
            if new_column in updated_df.columns:
                if column in updated_df.columns:
                    updated_df[column] = updated_df[new_column].combine_first(updated_df[column])
                else:
                    updated_df[column] = updated_df[new_column]
                updated_df.drop(columns=[new_column], inplace=True)
        updated_df.to_csv(Path(args.output), index=False, encoding="utf-8-sig")
        print(f"补数成功日期: {len(succeeded_dates)}")
    else:
        updated_df = existing_df
        print("没有新增成功日期")

    merged_failed_rows = [{"日期": date_text, "错误": "未处理"} for date_text in sorted(untouched_dates)] + failed_rows
    if merged_failed_rows:
        pd.DataFrame(merged_failed_rows).to_csv(Path(args.failed_dates), index=False, encoding="utf-8-sig")
        print(f"仍失败日期: {len(merged_failed_rows)}")
    else:
        pd.DataFrame(columns=["日期", "错误"]).to_csv(Path(args.failed_dates), index=False, encoding="utf-8-sig")
        print("失败清单已清空")


if __name__ == "__main__":
    main()
