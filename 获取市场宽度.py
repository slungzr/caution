from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from akshare.stock_feature.stock_a_indicator import headers as LEGU_HEADERS


BASE_DIR = Path(__file__).resolve().parent
HISTORY_CSV = BASE_DIR / "市场宽度历史.csv"
SNAPSHOT_JSON = BASE_DIR / "最新市场宽度.json"
MARKET_BREADTH_SYMBOL = "all"
MARKET_BREADTH_URL = (
    "https://www.legulegu.com/stockdata/member-ship/get-high-low-statistics/"
    f"{MARKET_BREADTH_SYMBOL}"
)


def parse_legu_date(value: object) -> pd.Timestamp:
    if isinstance(value, list) and len(value) >= 3:
        return pd.Timestamp(year=int(value[0]), month=int(value[1]), day=int(value[2]))
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return pd.to_datetime(int(value), unit="ms").normalize()
    return pd.to_datetime(value).normalize()


def read_history_if_exists(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            history_df = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"读取历史文件失败: {csv_path}")

    if history_df.empty:
        return history_df

    history_df["日期"] = pd.to_datetime(history_df["日期"], errors="coerce").dt.normalize()
    return history_df


def fetch_market_breadth() -> pd.DataFrame:
    response = requests.get(MARKET_BREADTH_URL, headers=LEGU_HEADERS, timeout=30)
    response.raise_for_status()
    raw_df = pd.DataFrame(response.json())
    if raw_df.empty:
        raise RuntimeError("市场宽度接口返回为空")

    breadth_df = raw_df.copy()
    breadth_df["日期"] = breadth_df["date"].apply(parse_legu_date)
    for source_column in ["close", "high20", "low20", "high60", "low60", "high120", "low120"]:
        breadth_df[source_column] = pd.to_numeric(breadth_df[source_column], errors="coerce")

    breadth_df = breadth_df.rename(
        columns={
            "close": "市场指数收盘",
            "high20": "市场20日新高数",
            "low20": "市场20日新低数",
            "high60": "市场60日新高数",
            "low60": "市场60日新低数",
            "high120": "市场120日新高数",
            "low120": "市场120日新低数",
        }
    )
    breadth_df["市场20日高低差"] = breadth_df["市场20日新高数"] - breadth_df["市场20日新低数"]
    breadth_df["市场60日高低差"] = breadth_df["市场60日新高数"] - breadth_df["市场60日新低数"]
    breadth_df["市场120日高低差"] = breadth_df["市场120日新高数"] - breadth_df["市场120日新低数"]

    keep_columns = [
        "日期",
        "市场指数收盘",
        "市场20日新高数",
        "市场20日新低数",
        "市场20日高低差",
        "市场60日新高数",
        "市场60日新低数",
        "市场60日高低差",
        "市场120日新高数",
        "市场120日新低数",
        "市场120日高低差",
    ]
    breadth_df = breadth_df[keep_columns]
    breadth_df = breadth_df.dropna(subset=["日期"]).drop_duplicates(subset=["日期"], keep="last")
    breadth_df = breadth_df.sort_values("日期").reset_index(drop=True)
    return breadth_df


def merge_history(existing_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        merged_df = latest_df.copy()
    else:
        merged_df = pd.concat([existing_df, latest_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["日期"], keep="last")

    merged_df = merged_df.sort_values("日期").reset_index(drop=True)
    return merged_df


def export_history(history_df: pd.DataFrame, csv_path: Path) -> None:
    export_df = history_df.copy()
    export_df["日期"] = pd.to_datetime(export_df["日期"]).dt.strftime("%Y-%m-%d")
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def export_snapshot(history_df: pd.DataFrame, snapshot_path: Path) -> dict[str, object]:
    latest_row = history_df.iloc[-1]
    latest_date = pd.Timestamp(latest_row["日期"]).strftime("%Y-%m-%d")
    diff20 = int(latest_row["市场20日高低差"])
    snapshot = {
        "日期": latest_date,
        "市场20日新高数": int(latest_row["市场20日新高数"]),
        "市场20日新低数": int(latest_row["市场20日新低数"]),
        "市场20日高低差": diff20,
        "开仓开关": "通过" if diff20 >= 0 else "不通过",
        "规则": "市场20日高低差 >= 0",
        "更新时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "数据来源": MARKET_BREADTH_URL,
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def main() -> None:
    fetched_df = fetch_market_breadth()
    history_df = merge_history(read_history_if_exists(HISTORY_CSV), fetched_df)
    export_history(history_df, HISTORY_CSV)
    snapshot = export_snapshot(history_df, SNAPSHOT_JSON)

    print(f"最新可用日期: {snapshot['日期']}")
    print(f"市场20日新高数: {snapshot['市场20日新高数']}")
    print(f"市场20日新低数: {snapshot['市场20日新低数']}")
    print(f"市场20日高低差: {snapshot['市场20日高低差']}")
    print(f"开仓开关: {snapshot['开仓开关']} (规则: {snapshot['规则']})")
    print(f"历史文件: {HISTORY_CSV}")
    print(f"最新快照: {SNAPSHOT_JSON}")


if __name__ == "__main__":
    main()
