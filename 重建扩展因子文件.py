from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "竞价爬升-20240504.csv"
OPENAPI_CACHE_DIR = BASE_DIR / "cache" / "wencai_openapi_factors"
UNIFIED_CACHE_DIR = BASE_DIR / "cache" / "wencai_unifiedwap_factors"
OUTPUT_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"读取失败: {csv_path}")


def normalize_stock_code(value: object) -> str:
    digits = re.findall(r"\d+", str(value))
    return "".join(digits)[-6:].zfill(6) if digits else ""


def load_cache_frames(cache_dir: Path) -> list[pd.DataFrame]:
    if not cache_dir.exists():
        return []
    frames: list[pd.DataFrame] = []
    for file_path in sorted(cache_dir.glob("*.csv")):
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if df.empty:
            continue
        df["日期"] = file_path.stem
        if "基础代码" not in df.columns:
            code_column = "股票代码" if "股票代码" in df.columns else "code"
            df["基础代码"] = df[code_column].apply(normalize_stock_code)
        else:
            df["基础代码"] = df["基础代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        frames.append(df)
    return frames


def main() -> None:
    source_df = read_csv_with_fallback(SOURCE_CSV)
    source_df["日期"] = source_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    source_df["基础代码"] = source_df["股票代码"].apply(normalize_stock_code)

    factor_frames = load_cache_frames(OPENAPI_CACHE_DIR) + load_cache_frames(UNIFIED_CACHE_DIR)
    if not factor_frames:
        raise RuntimeError("没有可用缓存可重建")

    factor_df = pd.concat(factor_frames, ignore_index=True)
    factor_df = factor_df.sort_values(["日期", "基础代码"]).drop_duplicates(subset=["日期", "基础代码"], keep="last")

    keep_columns = [column for column in factor_df.columns if column not in {"股票代码", "股票简称", "code"}]
    factor_df = factor_df[keep_columns].copy()

    rebuilt_df = source_df.merge(factor_df, on=["日期", "基础代码"], how="left")
    rebuilt_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    factor_dates = rebuilt_df.loc[rebuilt_df["竞价强度"].notna(), "日期"].nunique() if "竞价强度" in rebuilt_df.columns else 0
    print(f"重建完成: {OUTPUT_CSV}")
    print(f"因子覆盖交易日: {factor_dates}")


if __name__ == "__main__":
    main()
