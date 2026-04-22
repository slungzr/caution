from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from akshare.stock_feature.stock_a_indicator import headers as LEGU_HEADERS


BASE_DIR = Path(__file__).resolve().parent
HISTORY_CSV = BASE_DIR / "市场涨跌家数历史.csv"
SNAPSHOT_JSON = BASE_DIR / "最新市场涨跌家数.json"
API_PATH = "https://legulegu.com/stockdata/market-activity-trend-data"


def build_token(target_date: datetime) -> str:
    return hashlib.md5(target_date.strftime("%Y-%m-%d").encode("utf-8")).hexdigest()


def fetch_intraday_market_activity() -> pd.DataFrame:
    now = datetime.now()
    token = build_token(now)
    response = requests.get(f"{API_PATH}?token={token}", headers=LEGU_HEADERS, timeout=30)
    response.raise_for_status()
    raw = response.json()
    df = pd.DataFrame(raw)
    if df.empty:
        raise ValueError("未获取到市场涨跌家数数据")

    df["时间戳"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    df["日期"] = df["时间戳"].dt.normalize()
    numeric_columns = [
        "total",
        "totalUp",
        "totalDown",
        "priceStop",
        "pricePaused",
        "limitUp",
        "realLimitUp",
        "limitDown",
        "realLimitDown",
        "up0To3",
        "up3To5",
        "up5To7",
        "up7To10",
        "up10To20",
        "down0To3",
        "down3To5",
        "down5To7",
        "down7To10",
        "down10To20",
        "stLimitUp",
        "stLimitDown",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_daily_snapshot(intraday_df: pd.DataFrame) -> pd.DataFrame:
    latest = intraday_df.sort_values("时间戳").groupby("日期", as_index=False).tail(1).copy()
    latest["上涨家数"] = latest["totalUp"]
    latest["下跌家数"] = latest["totalDown"]
    latest["涨跌家数差"] = latest["上涨家数"] - latest["下跌家数"]
    latest["上涨占比"] = latest["上涨家数"] / latest["total"]
    latest["下跌占比"] = latest["下跌家数"] / latest["total"]
    latest["真实涨停家数"] = latest["realLimitUp"]
    latest["真实跌停家数"] = latest["realLimitDown"]
    latest["涨停家数"] = latest["limitUp"]
    latest["跌停家数"] = latest["limitDown"]
    latest["平盘家数"] = latest["priceStop"]
    latest["停牌家数"] = latest["pricePaused"]

    columns = [
        "日期",
        "时间戳",
        "total",
        "上涨家数",
        "下跌家数",
        "涨跌家数差",
        "上涨占比",
        "下跌占比",
        "涨停家数",
        "跌停家数",
        "真实涨停家数",
        "真实跌停家数",
        "平盘家数",
        "停牌家数",
    ]
    return latest[columns].rename(columns={"total": "总股票数"}).reset_index(drop=True)


def update_history(daily_df: pd.DataFrame) -> pd.DataFrame:
    if HISTORY_CSV.exists():
        history_df = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
        history_df["日期"] = pd.to_datetime(history_df["日期"]).dt.normalize()
        history_df["时间戳"] = pd.to_datetime(history_df["时间戳"])
        combined = pd.concat([history_df, daily_df], ignore_index=True)
        combined = combined.sort_values(["日期", "时间戳"]).drop_duplicates(subset=["日期"], keep="last")
    else:
        combined = daily_df.copy()

    combined = combined.sort_values("日期").reset_index(drop=True)
    export_df = combined.copy()
    export_df["日期"] = export_df["日期"].dt.strftime("%Y-%m-%d")
    export_df["时间戳"] = pd.to_datetime(export_df["时间戳"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    export_df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    return combined


def write_snapshot(latest_row: pd.Series) -> None:
    payload = {
        "日期": pd.Timestamp(latest_row["日期"]).strftime("%Y-%m-%d"),
        "时间戳": pd.Timestamp(latest_row["时间戳"]).strftime("%Y-%m-%d %H:%M:%S%z"),
        "总股票数": int(latest_row["总股票数"]),
        "上涨家数": int(latest_row["上涨家数"]),
        "下跌家数": int(latest_row["下跌家数"]),
        "涨跌家数差": int(latest_row["涨跌家数差"]),
        "上涨占比": round(float(latest_row["上涨占比"]), 6),
        "下跌占比": round(float(latest_row["下跌占比"]), 6),
        "涨停家数": int(latest_row["涨停家数"]),
        "跌停家数": int(latest_row["跌停家数"]),
        "真实涨停家数": int(latest_row["真实涨停家数"]),
        "真实跌停家数": int(latest_row["真实跌停家数"]),
        "平盘家数": int(latest_row["平盘家数"]),
        "停牌家数": int(latest_row["停牌家数"]),
    }
    pd.Series(payload).to_json(SNAPSHOT_JSON, force_ascii=False, indent=2)


def main() -> None:
    intraday_df = fetch_intraday_market_activity()
    daily_df = build_daily_snapshot(intraday_df)
    history_df = update_history(daily_df)
    latest_row = history_df.iloc[-1]
    write_snapshot(latest_row)

    print("市场涨跌家数更新完成")
    print(f"日期: {pd.Timestamp(latest_row['日期']).strftime('%Y-%m-%d')}")
    print(f"时间戳: {pd.Timestamp(latest_row['时间戳']).strftime('%Y-%m-%d %H:%M:%S%z')}")
    print(f"上涨家数: {int(latest_row['上涨家数'])}")
    print(f"下跌家数: {int(latest_row['下跌家数'])}")
    print(f"涨跌家数差: {int(latest_row['涨跌家数差'])}")
    print(f"上涨占比: {float(latest_row['上涨占比']):.2%}")
    print(f"真实涨停家数: {int(latest_row['真实涨停家数'])}")
    print(f"真实跌停家数: {int(latest_row['真实跌停家数'])}")


if __name__ == "__main__":
    main()
