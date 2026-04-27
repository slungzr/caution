from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_CSV = BASE_DIR / "竞价爬升-20240504.csv"
EXTENDED_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-至20260423.csv"
OUTPUT_CSV = BASE_DIR / "竞价爬升-20240504-补至20260423.csv"
DAILY_CACHE_DIR = BASE_DIR / "cache" / "daily_history"
TRADE_CALENDAR_CSV = BASE_DIR / "trade_calendar.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按原始竞价爬升 CSV 结构补齐数据")
    parser.add_argument("--original", default=str(ORIGINAL_CSV), help="原始 CSV")
    parser.add_argument("--extended", default=str(EXTENDED_CSV), help="补齐后的扩展因子 CSV")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="输出 CSV")
    parser.add_argument("--end-date", default="20260423", help="补齐截止日期 YYYYMMDD")
    return parser.parse_args()


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"读取失败: {csv_path}")


def normalize_stock_code(value: Any) -> str:
    digits = re.findall(r"\d+", str(value))
    return "".join(digits)[-6:].zfill(6) if digits else ""


def to_akshare_symbol(code: str) -> str:
    if code.startswith(("4", "8")):
        return f"bj{code}"
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def normalize_date_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def load_trade_dates() -> list[pd.Timestamp]:
    if TRADE_CALENDAR_CSV.exists():
        calendar_df = pd.read_csv(TRADE_CALENDAR_CSV, encoding="utf-8-sig")
    else:
        calendar_df = ak.tool_trade_date_hist_sina()
        calendar_df.to_csv(TRADE_CALENDAR_CSV, index=False, encoding="utf-8-sig")
    date_column = "trade_date" if "trade_date" in calendar_df.columns else calendar_df.columns[0]
    dates = pd.to_datetime(calendar_df[date_column], errors="coerce").dropna().dt.normalize()
    return sorted(pd.Timestamp(date) for date in dates.unique())


def build_next_trade_date_map(trade_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Timestamp]:
    return {
        pd.Timestamp(trade_dates[index]).normalize(): pd.Timestamp(trade_dates[index + 1]).normalize()
        for index in range(len(trade_dates) - 1)
    }


def normalize_daily_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    df = raw_df.rename(columns=rename_map).copy()
    keep_columns = [
        column
        for column in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
        if column in df.columns
    ]
    df = df[keep_columns].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
    df["prev_close"] = df["close"].shift(1)
    df["pct_chg"] = df["close"] / df["prev_close"] * 100 - 100
    return df.set_index("date")


def read_cached_history(code: str) -> pd.DataFrame:
    cache_path = DAILY_CACHE_DIR / f"{code}.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    return normalize_daily_history(pd.read_csv(cache_path, encoding="utf-8-sig"))


def fetch_history(code: str, end_date: pd.Timestamp) -> pd.DataFrame:
    cached_df = read_cached_history(code)
    if not cached_df.empty and pd.Timestamp(cached_df.index.max()) >= end_date:
        return cached_df

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            raw_df = ak.stock_zh_a_daily(
                symbol=to_akshare_symbol(code),
                start_date="19900101",
                end_date=end_date.strftime("%Y%m%d"),
                adjust="",
            )
            history_df = normalize_daily_history(raw_df)
            if not history_df.empty:
                DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                history_df.reset_index().to_csv(DAILY_CACHE_DIR / f"{code}.csv", index=False, encoding="utf-8-sig")
            return history_df
        except Exception as exc:  # pragma: no cover - 依赖网络
            last_error = exc
            time.sleep(attempt)
    print(f"日线拉取失败 {code}: {last_error}")
    return cached_df


def quote(history_df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.Series | None:
    if history_df.empty or trade_date not in history_df.index:
        return None
    row = history_df.loc[trade_date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def numeric_value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return float("nan")
    return pd.to_numeric(row[column], errors="coerce")


def fill_result_columns(
    df: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    next_trade_date_map: dict[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    out = df.copy()
    out["日期_dt"] = pd.to_datetime(out["日期"], format="%Y%m%d", errors="coerce").dt.normalize()
    for index, row in out.iterrows():
        code = str(row["基础代码"])
        trade_date = pd.Timestamp(row["日期_dt"])
        next_date = next_trade_date_map.get(trade_date)
        history_df = histories.get(code, pd.DataFrame())
        today_quote = quote(history_df, trade_date)
        next_quote = quote(history_df, next_date) if next_date is not None else None

        open_today = numeric_value(row, "开盘价:不复权今日")
        close_today = numeric_value(row, "收盘价:不复权今日")
        low_today = numeric_value(row, "最低价:不复权今日")

        if today_quote is not None:
            if pd.isna(open_today):
                open_today = float(today_quote.get("open", float("nan")))
                out.at[index, "开盘价:不复权今日"] = open_today
            if pd.isna(close_today):
                close_today = float(today_quote.get("close", float("nan")))
                out.at[index, "收盘价:不复权今日"] = close_today
            if pd.isna(low_today):
                low_today = float(today_quote.get("low", float("nan")))
                out.at[index, "最低价:不复权今日"] = low_today
            pct_today = float(today_quote.get("pct_chg", float("nan")))
            if pd.isna(numeric_value(row, "涨跌幅:前复权今日")) and not pd.isna(pct_today):
                out.at[index, "涨跌幅:前复权今日"] = pct_today

        if next_quote is not None:
            next_open = float(next_quote.get("open", float("nan")))
            next_close = float(next_quote.get("close", float("nan")))
            out.at[index, "开盘价:不复权明日"] = next_open
            out.at[index, "收盘价:不复权明日"] = next_close
            if not pd.isna(open_today) and open_today > 0:
                out.at[index, "开盘收益率"] = next_open / open_today - 1
                out.at[index, "收益率"] = next_close / open_today - 1

        if not pd.isna(open_today) and open_today > 0 and not pd.isna(low_today):
            out.at[index, "开盘后最大跌幅"] = low_today / open_today - 1

        if pd.isna(numeric_value(row, "实体涨跌幅今日")) and not pd.isna(open_today) and open_today > 0 and not pd.isna(close_today):
            out.at[index, "实体涨跌幅今日"] = (close_today / open_today - 1) * 100

        if pd.isna(numeric_value(row, "量比")):
            volume_ratio = numeric_value(row, "昨日前日成交量比")
            if not pd.isna(volume_ratio):
                out.at[index, "量比"] = volume_ratio

    return out.drop(columns=["日期_dt"])


def main() -> None:
    args = parse_args()
    end_date = pd.to_datetime(args.end_date, format="%Y%m%d").normalize()
    original_df = read_csv_with_fallback(Path(args.original))
    extended_df = read_csv_with_fallback(Path(args.extended))

    original_columns = list(original_df.columns)
    extended_df["日期"] = normalize_date_text(extended_df["日期"])
    extended_df["基础代码"] = extended_df["股票代码"].apply(normalize_stock_code)
    for column in original_columns:
        if column not in extended_df.columns:
            extended_df[column] = pd.NA

    original_df["日期"] = normalize_date_text(original_df["日期"])
    original_df["基础代码"] = original_df["股票代码"].apply(normalize_stock_code)
    original_max_date = pd.to_datetime(original_df["日期"], format="%Y%m%d", errors="coerce").max()

    append_df = extended_df[original_columns + ["基础代码", "昨日前日成交量比"]].copy()
    append_df["日期_dt"] = pd.to_datetime(append_df["日期"], format="%Y%m%d", errors="coerce").dt.normalize()
    append_df = append_df[(append_df["日期_dt"] > original_max_date) & (append_df["日期_dt"] <= end_date)].copy()
    if append_df.empty:
        rebuilt_df = original_df[original_columns].copy()
        rebuilt_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"无需新增，已输出: {args.output}")
        return

    trade_dates = load_trade_dates()
    next_trade_date_map = build_next_trade_date_map(trade_dates)
    max_needed_date = next_trade_date_map.get(end_date, end_date)
    codes = sorted(append_df["基础代码"].dropna().astype(str).unique().tolist())
    histories = {code: fetch_history(code, max_needed_date) for code in codes}
    append_df = append_df.drop(columns=["日期_dt"])
    append_df = fill_result_columns(append_df, histories, next_trade_date_map)

    rebuilt_df = pd.concat(
        [original_df[original_columns], append_df[original_columns]],
        ignore_index=True,
        sort=False,
    )
    rebuilt_df = rebuilt_df.sort_values(["日期", "个股热度排名昨日", "股票代码"], kind="stable").reset_index(drop=True)
    rebuilt_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    date_values = rebuilt_df["日期"].astype(str)
    print(f"输出完成: {args.output}")
    print(f"行数: {len(rebuilt_df)}")
    print(f"日期范围: {date_values.min()} - {date_values.max()}")
    print(f"交易日数: {date_values.nunique()}")


if __name__ == "__main__":
    main()
