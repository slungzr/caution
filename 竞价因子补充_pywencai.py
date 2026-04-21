from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import pywencai


BASE_DIR = Path(__file__).resolve().parent
EXISTING_FACTOR_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
FAILED_DATES_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-失败日期.csv"
OUTPUT_CSV = EXISTING_FACTOR_CSV
CACHE_DIR = BASE_DIR / "cache" / "wencai_pywencai_factors"

COOKIE = r'''other_uid=Ths_iwencai_Xuangu_7upib8nil3lgv9oqx1kb9k2g6mgo5dp3; cid=98ceb7f0b60eed590fd32d13d6b1d31e1776614455; ttype=WEB; user=MDpteF8xNzgyNTA0NDM6Ok5vbmU6NTAwOjE4ODI1MDQ0Mzo1LDEsNDA7NiwxLDQwOzcsMTExMTExMTExMTEwLDQwOzgsMTExMTAxMTEwMDAwMTExMTEwMDEwMDAwMDEwMDAwMDAsODk7MzMsMDAwMTAwMDAwMDAwLDg5OzM2LDEwMDExMTExMDAwMDExMDAxMDExMTExMSw4OTs0NiwwMDAwMTExMTEwMDAwMDExMTExMTExMTEsODk7NTEsMTEwMDAwMDAwMDAwMDAwMCw4OTs1OCwwMDAwMDAwMDAwMDAwMDAwMSw4OTs3OCwxLDg5Ozg3LDAwMDAwMDAwMDAwMDAwMDAwMDAxMDAwMCw4OTsxMTksMDAwMDAwMDAwMDAwMDAwMDAwMTAxMDAwMDAwMDAwMDAwMDAwMDAwMDAsODk7MTI1LDExLDg5OzEzMCwxMDEwMDAwMDAwMDAwLDg5OzQ0LDExLDQwOjE2Ojo6MTc4MjUwNDQzOjE3NzY2MTU4MjU6OjoxMzg1Nzc5ODYwOjYwNDgwMDowOjE4MWMxZjEyZjZiMWY5MTMyZTI1Y2QzMWRlMDAzZmFiNjpkZWZhdWx0XzU6MQ%3D%3D; userid=178250443; u_name=mx_178250443; escapename=mx_178250443; ticket=25753d75b907ca015075e2341da60a15; user_status=0; utk=b06b15e7c591b2bc4421bb6a0fdde37e; sess_tk=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6InNlc3NfdGtfMSIsImJ0eSI6InNlc3NfdGsifQ.eyJqdGkiOiJiNmZhMDNlMDFkZDM1Y2UyMzI5MTFmNmIyZmYxYzE4MTEiLCJpYXQiOjE3NzY2MTU4MjUsImV4cCI6MTc3NzIyMDYyNSwic3ViIjoiMTc4MjUwNDQzIiwiaXNzIjoidXBhc3MuaXdlbmNhaS5jb20iLCJhdWQiOiIyMDIwMTExODUyODg5MDcyIiwiYWN0Ijoib2ZjIiwiY3VocyI6IjAwOGU5YmU4MmE3OGJhY2UzZDc3MmU0ZDZlZGY0M2RhYjA0MDRlYzZiZWRiMzgxNzNlMDI3MjBkZjk0NmVjNGQifQ.1HVi-uFjytdjeP05K7eSZ5RTW7EjJqLdcLhV0cRiKWxlJ51Cw7gBY4nIMvxUxiEitTmnrKPfxK5VA-xlQW7Kvg; cuc=myv4ghgefxao; THSSESSID=014c470358e6299bb3024865ca; _clck=gcuw5r%7C2%7Cg5d%7C0%7C0; _clsk=kx4t9ur0z1gi%7C1776682887599%7C2%7C1%7C; v=A2HMaNWfrNWCtQCiOEB1oAPTcCZ-Dtc5_4B5F8M3X8PGA49YC17l0I_SialQ'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 pywencai 兜底补充竞价因子")
    parser.add_argument("--existing", default=str(EXISTING_FACTOR_CSV))
    parser.add_argument("--failed-dates", default=str(FAILED_DATES_CSV))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--date-limit", type=int, default=1)
    parser.add_argument("--sleep-query", type=float, default=3.5)
    parser.add_argument("--sleep-date", type=float, default=5.0)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"读取失败: {csv_path}")


def date_to_cn(date_text: str) -> str:
    return f"{date_text[:4]}年{int(date_text[4:6])}月{int(date_text[6:8])}日"


def normalize_stock_code(value: object) -> str:
    digits = re.findall(r"\d+", str(value))
    return "".join(digits)[-6:].zfill(6) if digits else ""


def pick_column(columns: list[str], keyword: str) -> str | None:
    exact = [column for column in columns if column == keyword]
    if exact:
        return exact[0]
    matches = [column for column in columns if keyword in column]
    if not matches:
        return None
    matches.sort(key=len)
    return matches[0]


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for keyword, target in [
        ("股票代码", "股票代码"),
        ("股票简称", "股票简称"),
        ("竞价强度", "竞价强度"),
        ("竞价金额", "竞价匹配金额_openapi"),
        ("竞价未匹配金额", "竞价未匹配金额"),
        ("分时换手率", "竞价换手率_openapi"),
        ("集合竞价评级", "集合竞价评级_unifiedwap"),
        ("个股热度排名", "个股热度排名_openapi"),
    ]:
        source = pick_column(list(df.columns), keyword)
        if source is not None:
            rename_map[source] = target
    out = df.rename(columns=rename_map).copy()
    code_column = "股票代码" if "股票代码" in out.columns else pick_column(list(out.columns), "股票代码") or "code"
    if code_column in out.columns and code_column != "股票代码":
        out.rename(columns={code_column: "股票代码"}, inplace=True)
    if "股票简称" not in out.columns:
        out["股票简称"] = ""
    out["基础代码"] = out["股票代码"].apply(normalize_stock_code)
    keep = [c for c in ["股票代码","股票简称","基础代码","竞价强度","竞价匹配金额_openapi","竞价未匹配金额","竞价换手率_openapi","集合竞价评级_unifiedwap","个股热度排名_openapi"] if c in out.columns]
    return out[keep].copy()


def query_date(date_text: str, prev_date_text: str, refresh_cache: bool, sleep_query: float) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{date_text}.csv"
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        if not cached.empty:
            return cached

    queries = [
        f"{date_to_cn(date_text)}竞价强度 *1，{date_to_cn(prev_date_text)}个股热度排名前100",
        f"{date_to_cn(date_text)}竞价金额，{date_to_cn(prev_date_text)}个股热度排名前100",
        f"{date_to_cn(date_text)}竞价换手率，{date_to_cn(prev_date_text)}个股热度排名前100",
    ]
    merged = None
    for query in queries:
        df = pywencai.get(query=query, cookie=COOKIE)
        if df is None or getattr(df, 'empty', False):
            raise RuntimeError(f"pywencai 空结果: {query}")
        df = standardize(df)
        merged = df if merged is None else pd.merge(merged, df, on=[c for c in ["股票代码","基础代码"] if c in merged.columns and c in df.columns][0], how="outer", suffixes=("", "_drop"))
        if merged is not None:
            merged = merged.filter(regex=r"^(?!.*_drop$)")
        time.sleep(max(0.0, sleep_query))
    merged.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return merged


def main() -> None:
    args = parse_args()
    existing_df = read_csv_with_fallback(Path(args.existing))
    existing_df["日期"] = existing_df["日期"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    existing_df["基础代码"] = existing_df["基础代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    failed_df = read_csv_with_fallback(Path(args.failed_dates))
    target_dates = failed_df["日期"].astype(str).tolist()[: args.date_limit]
    all_dates = sorted(existing_df["日期"].unique().tolist())
    prev_map = {all_dates[i]: all_dates[i-1] for i in range(1, len(all_dates))}

    success_dates = []
    failed_rows = []
    frames = []
    for date_text in target_dates:
        print(f"补 {date_text}")
        prev = prev_map.get(date_text)
        if prev is None:
            failed_rows.append({"日期": date_text, "错误": "缺少上一交易日"})
            continue
        try:
            df = query_date(date_text, prev, args.refresh_cache, args.sleep_query)
            df["日期"] = date_text
            frames.append(df)
            success_dates.append(date_text)
        except Exception as exc:
            failed_rows.append({"日期": date_text, "错误": str(exc)})
        time.sleep(max(0.0, args.sleep_date))

    if frames:
        factor_all = pd.concat(frames, ignore_index=True)
        factor_columns = [c for c in factor_all.columns if c not in {"日期", "基础代码"}]
        untouched = existing_df.loc[~existing_df["日期"].isin(success_dates)].copy()
        target = existing_df.loc[existing_df["日期"].isin(success_dates)].copy().drop(columns=[c for c in factor_columns if c in existing_df.columns], errors="ignore")
        target = target.merge(factor_all, on=["日期", "基础代码"], how="left")
        updated = pd.concat([untouched, target], ignore_index=True).sort_values(["日期", "基础代码"], kind="stable")
        updated.to_csv(Path(args.output), index=False, encoding="utf-8-sig")

    untouched_dates = failed_df["日期"].astype(str).tolist()[args.date_limit:]
    remaining = [{"日期": d, "错误": "未处理"} for d in untouched_dates] + failed_rows
    pd.DataFrame(remaining).to_csv(Path(args.failed_dates), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
