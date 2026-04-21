from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "竞价爬升-20240504.csv"
FACTOR_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
FAILED_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-失败日期.csv"
START_DATE = "20260108"


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"读取失败: {csv_path}")


def main() -> None:
    source_df = read_csv_with_fallback(SOURCE_CSV)
    factor_df = read_csv_with_fallback(FACTOR_CSV)
    source_dates = sorted(source_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).unique().tolist())
    calendar_df = ak.tool_trade_date_hist_sina()
    trading_dates = set(pd.to_datetime(calendar_df["trade_date"]).dt.strftime("%Y%m%d").tolist())
    source_dates = [date for date in source_dates if date in trading_dates]
    factor_df["日期"] = factor_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    if "竞价强度" in factor_df.columns:
        factor_dates = set(factor_df.loc[factor_df["竞价强度"].notna(), "日期"].tolist())
    else:
        factor_dates = set()

    missing_dates = [date for date in source_dates if date >= START_DATE and date not in factor_dates]
    pd.DataFrame([{"日期": date, "错误": "未处理"} for date in missing_dates]).to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
    print(f"missing_dates={len(missing_dates)}")
    if missing_dates:
        print(f"first={missing_dates[0]}")
        print(f"last={missing_dates[-1]}")


if __name__ == "__main__":
    main()
