from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OPERATE_SCRIPT = BASE_DIR / "生成今日操作清单.py"
DEFAULT_INPUT = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
DEFAULT_OUTPUT = BASE_DIR / "竞价爬升-20240504-扩展因子-至20260423.csv"
CACHE_DIR = BASE_DIR / "cache" / "wencai_history_auction_signal"
FAILED_OUTPUT = BASE_DIR / "竞价爬升-20240504-扩展因子-补齐失败日期.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补齐竞价爬升扩展因子到指定日期")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="已有扩展因子 CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="补齐后的输出 CSV")
    parser.add_argument("--start-date", default="", help="起始日期 YYYYMMDD，默认从已有数据下一交易日开始")
    parser.add_argument("--end-date", default="20260423", help="截止日期 YYYYMMDD")
    parser.add_argument("--sleep", type=float, default=1.2, help="每个交易日之间暂停秒数")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略本地补齐缓存重新查询")
    return parser.parse_args()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"读取失败: {csv_path}")


def parse_yyyymmdd(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%Y%m%d").normalize()


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["日期"] = out["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return out


def get_target_trade_dates(
    operate_mod,
    existing_df: pd.DataFrame,
    start_date_text: str,
    end_date_text: str,
) -> list[pd.Timestamp]:
    calendar_df = operate_mod.load_trade_calendar()
    end_date = parse_yyyymmdd(end_date_text)
    if start_date_text:
        start_date = parse_yyyymmdd(start_date_text)
    else:
        existing_dates = pd.to_datetime(existing_df["日期"], format="%Y%m%d", errors="coerce").dropna()
        if existing_dates.empty:
            raise RuntimeError("已有扩展因子缺少有效日期，请显式传入 --start-date")
        start_date = existing_dates.max() + pd.Timedelta(days=1)

    dates = calendar_df.loc[
        (calendar_df["trade_date"] >= start_date) & (calendar_df["trade_date"] <= end_date),
        "trade_date",
    ].tolist()
    return [pd.Timestamp(date).normalize() for date in dates]


def build_prev_maps(operate_mod) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]:
    calendar_df = operate_mod.load_trade_calendar()
    dates = [pd.Timestamp(date).normalize() for date in calendar_df["trade_date"].tolist()]
    prev_map: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for index in range(2, len(dates)):
        prev_map[dates[index]] = (dates[index - 1], dates[index - 2])
    return prev_map


def build_queries(operate_mod, today_ts: pd.Timestamp, prev_ts: pd.Timestamp, prev2_ts: pd.Timestamp) -> dict[str, str]:
    queries = operate_mod.build_queries(today_ts, prev_ts, prev2_ts)
    today_cn = operate_mod.cn_date(today_ts)
    prev_cn = operate_mod.cn_date(prev_ts)
    queries["amount"] = (
        f"{today_cn}竞价强度，{today_cn}竞价金额，{today_cn}竞价未匹配金额，"
        f"{today_cn}竞价换手率，{today_cn}竞价量比，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    return queries


def pick_column(columns: list[str], keywords: list[str]) -> str | None:
    for keyword in keywords:
        exact = [column for column in columns if column == keyword]
        if exact:
            return exact[0]
    for keyword in keywords:
        matched = [column for column in columns if keyword in column]
        if matched:
            matched.sort(key=len)
            return matched[0]
    return None


def standardize_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    columns = list(out.columns)
    extra_mapping = {
        "竞价强度": ["竞价强度"],
        "竞价换手率_openapi": ["竞价换手率_openapi", "竞价换手率今日", "竞价换手率"],
        "竞价量比_openapi": ["竞价量比_openapi", "竞价量比今日", "竞价量比"],
        "个股热度排名_openapi": ["个股热度排名_openapi", "个股热度排名昨日", "个股热度排名"],
    }
    for target, keywords in extra_mapping.items():
        if target in out.columns and out[target].notna().any():
            continue
        source = pick_column(columns, keywords)
        if source is not None:
            out[target] = out[source]

    for column in extra_mapping:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "竞价强度" not in out.columns:
        out["竞价强度"] = pd.NA
    if "竞价匹配金额_openapi" in out.columns:
        amount = pd.to_numeric(out["竞价匹配金额_openapi"], errors="coerce")
    else:
        amount = pd.Series(pd.NA, index=out.index, dtype="Float64")
    strength = pd.to_numeric(out["竞价强度"], errors="coerce")
    # 当前正式基线不使用竞价强度数值，只沿用旧加载逻辑要求非空。
    out["竞价强度"] = strength.mask(strength.isna() & amount.notna(), 1.0)
    return out


def fetch_signal_for_date(
    operate_mod,
    trade_date: pd.Timestamp,
    prev_date: pd.Timestamp,
    prev2_date: pd.Timestamp,
    cookies: list[str],
    refresh_cache: bool,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{trade_date.strftime('%Y%m%d')}.csv"
    meta_path = CACHE_DIR / f"{trade_date.strftime('%Y%m%d')}.json"
    if cache_path.exists() and not refresh_cache:
        cached_df = pd.read_csv(cache_path, encoding="utf-8-sig")
        if not cached_df.empty:
            return standardize_extra_columns(cached_df)

    queries = build_queries(operate_mod, trade_date, prev_date, prev2_date)
    date_map = {
        operate_mod.date_token(trade_date): "今日",
        operate_mod.date_token(prev_date): "昨日",
        operate_mod.date_token(prev2_date): "前日",
    }

    frames: list[pd.DataFrame] = []
    for label in ["base", "detail", "amount"]:
        frame = operate_mod.query_wencai(queries[label], cookies)
        standardized = operate_mod.standardize_frame(frame, date_map)
        standardized = standardize_extra_columns(standardized)
        frames.append(standardized)

    merged = operate_mod.merge_frames(frames)
    if merged.empty:
        return merged
    merged = standardize_extra_columns(merged)
    merged = operate_mod.compute_factors(merged)
    merged["日期"] = trade_date.strftime("%Y%m%d")
    merged["个股热度排名_openapi"] = pd.to_numeric(
        merged.get("个股热度排名_openapi", merged.get("个股热度排名昨日")),
        errors="coerce",
    )

    merged.to_csv(cache_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def align_and_concat(existing_df: pd.DataFrame, append_df: pd.DataFrame) -> pd.DataFrame:
    if append_df.empty:
        return existing_df.copy()
    columns = list(existing_df.columns)
    extra_columns = [column for column in append_df.columns if column not in columns]
    for column in columns:
        if column not in append_df.columns:
            append_df[column] = pd.NA
    combined = pd.concat(
        [existing_df[columns], append_df[columns + extra_columns]],
        ignore_index=True,
        sort=False,
    )
    combined = normalize_date_column(combined)
    if "基础代码" in combined.columns:
        combined["基础代码"] = combined["基础代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    combined = combined.sort_values(["日期", "基础代码"], kind="stable")
    combined = combined.drop_duplicates(subset=["日期", "基础代码"], keep="last").reset_index(drop=True)
    return combined


def main() -> None:
    args = parse_args()
    operate_mod = load_module("operate_list_for_fill", OPERATE_SCRIPT)
    existing_df = normalize_date_column(read_csv_with_fallback(Path(args.input)))
    target_dates = get_target_trade_dates(operate_mod, existing_df, args.start_date, args.end_date)
    prev_map = build_prev_maps(operate_mod)
    cookies = operate_mod.resolve_cookies()

    print(f"已有数据最大日期: {existing_df['日期'].max()}")
    print(f"目标截止日期: {args.end_date}")
    print(f"待补交易日数: {len(target_dates)}")
    if not target_dates:
        Path(args.output).write_text("", encoding="utf-8")
        existing_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"无需补齐，已输出: {args.output}")
        return

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, trade_date in enumerate(target_dates, start=1):
        date_text = trade_date.strftime("%Y%m%d")
        print(f"[{index}/{len(target_dates)}] 补齐 {date_text}")
        prev_dates = prev_map.get(trade_date)
        if prev_dates is None:
            failures.append({"日期": date_text, "错误": "缺少上一/前二交易日"})
            continue
        try:
            day_df = fetch_signal_for_date(
                operate_mod,
                trade_date,
                prev_dates[0],
                prev_dates[1],
                cookies,
                args.refresh_cache,
            )
            if day_df.empty:
                print(f"  {date_text} 无竞价爬升信号")
                continue
            print(f"  {date_text} 拉取 {len(day_df)} 行")
            frames.append(day_df)
        except Exception as exc:
            failures.append({"日期": date_text, "错误": str(exc)})
            print(f"  {date_text} 失败: {exc}")
        time.sleep(max(0.0, args.sleep))

    append_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_df = align_and_concat(existing_df, append_df)
    combined_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(FAILED_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"输出完成: {args.output}")
    print(f"总行数: {len(combined_df)}")
    print(f"最大日期: {combined_df['日期'].max()}")
    print(f"失败日期数: {len(failures)}")
    if failures:
        print(pd.DataFrame(failures).to_string(index=False))


if __name__ == "__main__":
    main()
