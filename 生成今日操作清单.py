from __future__ import annotations

import json
import os
import re
import time
import argparse
import importlib.util
import logging
import sys
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import pywencai
import requests

from wencai_direct import fetch_query_dataframe, load_cookie as load_legacy_cookie


BASE_DIR = Path(__file__).resolve().parent
TRADE_CALENDAR_CSV = BASE_DIR / "trade_calendar.csv"
COOKIE_FILE = BASE_DIR / "wencai_cookie.txt"
PYWENCAI_SCRIPT = BASE_DIR / "竞价因子补充_pywencai.py"
UNIFIEDWAP_SCRIPT = BASE_DIR / "wencai_unifiedwap.py"
MARKET_SNAPSHOT_JSON = BASE_DIR / "最新市场宽度.json"
MARKET_HISTORY_CSV = BASE_DIR / "市场宽度历史.csv"
MARKET_WIDTH_SCRIPT = BASE_DIR / "获取市场宽度.py"
OUTPUT_PREFIX = BASE_DIR / "今日操作清单"
TOP_N = 2
INDUSTRY_CHANGE_MIN = 0.0
PREV_BODY_MIN = 0.0
SECTOR_EXPLORER_SCRIPT = BASE_DIR / "竞价行业联动探索.py"
OPEN_SECTOR_EXPLORER_SCRIPT = BASE_DIR / "竞价行业开盘联动探索.py"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
PUSHPLUS_TOKEN = "e60d2f5d230f42739c52712203b9eb93"
RUN_LOG = BASE_DIR / "自动化运行日志.log"

logging.basicConfig(
    filename=RUN_LOG,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9:25 后生成今日操作清单")
    parser.add_argument(
        "--trade-date",
        help="指定交易日期，格式 YYYYMMDD，用于历史回放或盘后验证",
    )
    parser.add_argument("--push", action="store_true", help="历史回放时也发送 PushPlus 推送")
    parser.add_argument("--no-push", action="store_true", help="本次运行不发送 PushPlus 推送")
    return parser.parse_args()


def read_snapshot(snapshot_path: Path) -> dict[str, Any]:
    if not snapshot_path.exists():
        raise FileNotFoundError(f"未找到市场开关快照: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def refresh_market_snapshot() -> None:
    if not MARKET_WIDTH_SCRIPT.exists():
        raise FileNotFoundError(f"未找到市场宽度刷新脚本: {MARKET_WIDTH_SCRIPT}")
    spec = importlib.util.spec_from_file_location("market_width_refresh", MARKET_WIDTH_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载市场宽度刷新脚本: {MARKET_WIDTH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fetched_df = module.fetch_market_breadth()
    history_df = module.merge_history(module.read_history_if_exists(MARKET_HISTORY_CSV), fetched_df)
    try:
        module.export_history(history_df, MARKET_HISTORY_CSV)
    except PermissionError as exc:
        logging.warning("市场宽度历史文件写入失败，继续更新最新快照: %s", exc)
    module.export_snapshot(history_df, MARKET_SNAPSHOT_JSON)


def ensure_fresh_snapshot(snapshot: dict[str, Any], required_date: pd.Timestamp | None) -> dict[str, Any]:
    if required_date is None:
        return snapshot
    snapshot_date = pd.Timestamp(snapshot["日期"]).normalize()
    required_date = required_date.normalize()
    if snapshot_date >= required_date:
        return snapshot

    logging.info(
        "市场快照过期，尝试刷新 snapshot_date=%s required_date=%s",
        snapshot_date.strftime("%Y-%m-%d"),
        required_date.strftime("%Y-%m-%d"),
    )
    refresh_market_snapshot()
    snapshot = read_snapshot(MARKET_SNAPSHOT_JSON)
    snapshot_date = pd.Timestamp(snapshot["日期"]).normalize()
    if snapshot_date < required_date:
        raise RuntimeError(
            f"市场宽度快照过期: 当前 {snapshot_date.strftime('%Y-%m-%d')}，"
            f"需要至少 {required_date.strftime('%Y-%m-%d')}。请检查 获取市场宽度.py 或数据源。"
        )
    return snapshot


def read_market_snapshot(
    trade_date: pd.Timestamp,
    historical_replay: bool,
    required_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if not historical_replay:
        return ensure_fresh_snapshot(read_snapshot(MARKET_SNAPSHOT_JSON), required_date)

    if not MARKET_HISTORY_CSV.exists():
        raise FileNotFoundError(f"未找到市场宽度历史文件: {MARKET_HISTORY_CSV}")

    history_df = pd.read_csv(MARKET_HISTORY_CSV, encoding="utf-8-sig")
    if "日期" not in history_df.columns or "市场20日高低差" not in history_df.columns:
        raise RuntimeError(f"市场宽度历史文件缺少必要字段: {MARKET_HISTORY_CSV}")

    history_df["日期"] = pd.to_datetime(history_df["日期"], errors="coerce").dt.normalize()
    target_date = trade_date.normalize()
    matched = history_df[history_df["日期"] == target_date].copy()
    if matched.empty:
        raise RuntimeError(
            f"市场宽度历史缺少 {target_date.strftime('%Y-%m-%d')}，请先运行 获取市场宽度.py 补齐历史数据"
        )

    row = matched.iloc[-1]
    diff20 = int(pd.to_numeric(row["市场20日高低差"], errors="raise"))
    snapshot = {
        "日期": target_date.strftime("%Y-%m-%d"),
        "市场20日高低差": diff20,
        "开仓开关": "通过" if diff20 >= 0 else "不通过",
        "规则": "市场20日高低差 >= 0",
        "数据来源": str(MARKET_HISTORY_CSV),
    }
    for column in ["市场20日新高数", "市场20日新低数"]:
        if column in row.index and pd.notna(row[column]):
            snapshot[column] = int(pd.to_numeric(row[column], errors="raise"))
    return snapshot


def load_trade_calendar() -> pd.DataFrame:
    if TRADE_CALENDAR_CSV.exists():
        df = pd.read_csv(TRADE_CALENDAR_CSV, encoding="utf-8-sig")
    else:
        df = ak.tool_trade_date_hist_sina()
        df.to_csv(TRADE_CALENDAR_CSV, index=False, encoding="utf-8-sig")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    return df.sort_values("trade_date").reset_index(drop=True)


def pick_trade_dates(today_value: date | None = None) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    today_ts = pd.Timestamp(today_value or date.today()).normalize()
    calendar_df = load_trade_calendar()
    valid_dates = calendar_df.loc[calendar_df["trade_date"] <= today_ts, "trade_date"].tolist()
    if len(valid_dates) < 3:
        raise RuntimeError("交易日历不足，无法取得今日/昨日/前日")
    return valid_dates[-1], valid_dates[-2], valid_dates[-3]


def cn_date(ts: pd.Timestamp) -> str:
    return f"{ts.month}月{ts.day}日"


def date_token(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def normalize_stock_code(value: Any) -> str:
    digits = re.findall(r"\d+", str(value))
    return "".join(digits)[-6:].zfill(6) if digits else ""


def try_read_cookie_from_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def extract_cookie_from_python(path: Path, variable_name: str) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    patterns = [
        rf"{variable_name}\s*=\s*r'''(.*?)'''",
        rf'{variable_name}\s*=\s*r"""(.*?)"""',
        rf"{variable_name}\s*=\s*'([^']+)'",
        rf'{variable_name}\s*=\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.S)
        if match:
            cookie = match.group(1).strip()
            if cookie:
                return cookie
    return None


def resolve_cookies() -> list[str]:
    cookies: list[str] = []
    candidates = [
        os.environ.get("WENCAI_COOKIE"),
        try_read_cookie_from_file(COOKIE_FILE),
        extract_cookie_from_python(PYWENCAI_SCRIPT, "COOKIE"),
        extract_cookie_from_python(UNIFIEDWAP_SCRIPT, "Cookie"),
    ]
    for candidate in candidates:
        if candidate and candidate not in cookies:
            cookies.append(candidate)

    for loader in [load_legacy_cookie]:
        try:
            cookie = loader()
            if cookie and cookie not in cookies:
                cookies.append(cookie)
        except Exception:
            continue

    if not cookies:
        raise RuntimeError("未找到可用问财 cookie，请更新 wencai_cookie.txt 或设置 WENCAI_COOKIE")
    return cookies


def load_sector_module():
    if not SECTOR_EXPLORER_SCRIPT.exists():
        raise FileNotFoundError(f"未找到行业联动脚本: {SECTOR_EXPLORER_SCRIPT}")
    spec = importlib.util.spec_from_file_location("sector_explorer", SECTOR_EXPLORER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载行业联动脚本: {SECTOR_EXPLORER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_open_sector_module():
    if not OPEN_SECTOR_EXPLORER_SCRIPT.exists():
        raise FileNotFoundError(f"未找到行业开盘联动脚本: {OPEN_SECTOR_EXPLORER_SCRIPT}")
    spec = importlib.util.spec_from_file_location("open_sector_explorer", OPEN_SECTOR_EXPLORER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载行业开盘联动脚本: {OPEN_SECTOR_EXPLORER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def query_wencai(question: str, cookies: list[str], pause_seconds: float = 1.5) -> pd.DataFrame:
    errors: list[str] = []
    for index, cookie in enumerate(cookies, start=1):
        try:
            df = fetch_query_dataframe(question, cookie)
            if df is not None and not df.empty:
                time.sleep(max(0.0, pause_seconds))
                return df
            errors.append(f"cookie{index}:direct_empty")
        except Exception as exc:
            errors.append(f"cookie{index}:direct={exc}")

        try:
            df = pywencai.get(query=question, cookie=cookie)
            if isinstance(df, pd.DataFrame) and not df.empty:
                time.sleep(max(0.0, pause_seconds))
                return df
            errors.append(f"cookie{index}:pywencai_empty")
        except Exception as exc:
            errors.append(f"cookie{index}:pywencai={exc}")

    raise RuntimeError(f"问财查询失败: {'; '.join(errors)}")


def fetch_sector_snapshot(
    sector_mod,
    trade_date: pd.Timestamp,
    historical_replay: bool,
) -> tuple[pd.DataFrame, str]:
    session = sector_mod.create_session()
    if historical_replay:
        open_sector_mod = load_open_sector_module()
        first_index_df = sector_mod.fetch_first_level_index_list(session)
        sector_df = open_sector_mod.fetch_first_level_open_history(
            open_sector_mod.create_session(),
            first_index_df,
        ).copy()
        sector_df = sector_df[sector_df["日期"] == trade_date.normalize()].copy()
        sector_df = sector_df.rename(
            columns={
                "申万一级行业开盘涨幅": "申万一级行业涨跌幅",
                "申万一级行业开盘涨幅排名": "申万一级行业涨跌幅排名",
            }
        )
        keep_columns = [
            "日期",
            "申万一级行业代码",
            "申万一级行业",
            "申万一级行业涨跌幅",
            "申万一级行业涨跌幅排名",
        ]
        return sector_df[keep_columns], "申万一级行业开盘涨幅（仅用于历史复盘）"

    cache_path = sector_mod.CACHE_DIR / f"sw_first_realtime_{date_token(trade_date)}.csv"
    try:
        response = sector_mod.request_with_retry(
            session,
            "https://www.swsresearch.com/institute-sw/api/index_publish/current/",
            params={
                "page": "1",
                "page_size": "100",
                "indextype": "一级行业",
            },
        )
        realtime_df = pd.DataFrame(response.json()["data"]["results"]).copy()
        realtime_df = realtime_df.rename(
            columns={
                "swindexcode": "申万一级行业代码",
                "swindexname": "申万一级行业",
            }
        )
        realtime_df["日期"] = trade_date.normalize()
        realtime_df["申万一级行业昨收"] = pd.to_numeric(realtime_df["l3"], errors="coerce")
        realtime_df["申万一级行业今开"] = pd.to_numeric(realtime_df["l4"], errors="coerce")
        realtime_df["申万一级行业成交额"] = pd.to_numeric(realtime_df["l5"], errors="coerce")
        realtime_df["申万一级行业涨跌幅"] = (
            realtime_df["申万一级行业今开"] / realtime_df["申万一级行业昨收"] - 1
        )
        realtime_df["申万一级行业涨跌幅排名"] = realtime_df["申万一级行业涨跌幅"].rank(
            ascending=False,
            method="min",
        )
        keep_columns = [
            "日期",
            "申万一级行业代码",
            "申万一级行业",
            "申万一级行业涨跌幅",
            "申万一级行业涨跌幅排名",
            "申万一级行业成交额",
        ]
        realtime_df = realtime_df[keep_columns].sort_values(
            ["申万一级行业涨跌幅排名", "申万一级行业代码"],
            ascending=[True, True],
            kind="stable",
        )
        sector_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        realtime_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return realtime_df, "申万一级行业开盘涨幅（实时）"
    except Exception:
        if not cache_path.exists():
            raise
        cached_df = pd.read_csv(cache_path, encoding="utf-8-sig")
        cached_df["日期"] = sector_mod.normalize_trade_date(cached_df["日期"])
        for column in ["申万一级行业涨跌幅", "申万一级行业涨跌幅排名"]:
            if column in cached_df.columns:
                cached_df[column] = pd.to_numeric(cached_df[column], errors="coerce")
        return cached_df, "申万一级行业开盘涨幅（缓存）"


def attach_sector_context(
    df: pd.DataFrame,
    trade_date: pd.Timestamp,
    historical_replay: bool,
) -> tuple[pd.DataFrame, str]:
    sector_mod = load_sector_module()
    session = sector_mod.create_session()
    first_index_df = sector_mod.fetch_first_level_index_list(session)
    first_component_df = sector_mod.fetch_first_level_components(session, first_index_df)
    industry_history_df = sector_mod.fetch_stock_industry_history(session)
    histcode_map_df = sector_mod.build_histcode_to_first_map(industry_history_df, first_component_df)

    annotated_input = df.copy()
    annotated_input["日期"] = trade_date.normalize()
    annotated_df = sector_mod.annotate_first_industry(
        annotated_input,
        industry_history_df,
        histcode_map_df,
    )
    if "申万一级行业代码" in annotated_df.columns:
        annotated_df["申万一级行业代码"] = annotated_df["申万一级行业代码"].astype(str)

    sector_df, sector_source = fetch_sector_snapshot(sector_mod, trade_date, historical_replay)
    if "申万一级行业代码" in sector_df.columns:
        sector_df["申万一级行业代码"] = sector_df["申万一级行业代码"].astype(str)
    merged_df = annotated_df.merge(
        sector_df,
        on=["日期", "申万一级行业代码", "申万一级行业"],
        how="left",
    )
    return merged_df, sector_source


def build_queries(today_ts: pd.Timestamp, prev_ts: pd.Timestamp, prev2_ts: pd.Timestamp) -> dict[str, str]:
    today_cn = cn_date(today_ts)
    prev_cn = cn_date(prev_ts)
    prev2_cn = cn_date(prev2_ts)

    base_query = (
        f"{today_cn}9点25分最低价>{today_cn}9点24分最高价，"
        f"{today_cn}9点24分最低价>={today_cn}9点23分最高价，"
        f"{today_cn}9点23分最低价>={today_cn}9点22分最高价，"
        f"{today_cn}9点22分最低价>={today_cn}9点21分最高价，"
        f"{today_cn}9点21分最低价>={today_cn}9点20分最高价，"
        f"{today_cn}竞价涨幅，{today_cn}竞价换手率，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    detail_query = (
        f"{today_cn}开盘价:不复权，{prev_cn}成交量，{prev2_cn}成交量，"
        f"{prev_cn}实体涨跌幅，{prev2_cn}实体涨跌幅，"
        f"{prev_cn}个股热度排名，{prev_cn}连续涨停天数，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    amount_query = (
        f"{today_cn}竞价金额，{today_cn}竞价未匹配金额，{prev_cn}成交金额，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    return {"base": base_query, "detail": detail_query, "amount": amount_query}


def replace_date_markers(columns: pd.Index, mapping: dict[str, str]) -> pd.Index:
    updated = columns.astype(str)
    for token, label in mapping.items():
        updated = updated.str.replace(rf"\[{token}\]", label, regex=True)
    return updated


def pick_column(columns: list[str], keywords: list[str]) -> str | None:
    for keyword in keywords:
        exact = [column for column in columns if column == keyword]
        if exact:
            return exact[0]
    for keyword in keywords:
        partial = [column for column in columns if keyword in column]
        if partial:
            partial.sort(key=len)
            return partial[0]
    return None


def standardize_frame(df: pd.DataFrame, date_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = replace_date_markers(out.columns, date_map)
    rename_map: dict[str, str] = {}
    columns = list(out.columns)
    keyword_mapping = {
        "股票代码": ["股票代码", "code", "证券代码"],
        "股票简称": ["股票简称", "简称", "股票简称最新"],
        "竞价涨幅今日": ["竞价涨幅今日", "竞价涨幅"],
        "竞价换手率今日": ["竞价换手率今日", "竞价换手率", "分时换手率今日"],
        "竞价匹配金额_openapi": ["竞价匹配金额_openapi", "竞价金额今日", "竞价金额"],
        "竞价未匹配金额": ["竞价未匹配金额"],
        "开盘价:不复权今日": ["开盘价:不复权今日", "开盘价今日", "开盘价:不复权"],
        "成交金额昨日": ["成交金额昨日", "成交额昨日", "成交金额", "成交额"],
        "成交量昨日": ["成交量昨日", "成交量"],
        "成交量前日": ["成交量前日"],
        "实体涨跌幅昨日": ["实体涨跌幅昨日", "实体涨跌幅"],
        "实体涨跌幅前日": ["实体涨跌幅前日"],
        "个股热度排名昨日": ["个股热度排名昨日", "个股热度排名", "个股热度排名前100"],
        "连续涨停天数昨日": ["连续涨停天数昨日", "连续涨停天数"],
    }
    for target, keywords in keyword_mapping.items():
        source = pick_column(columns, keywords)
        if source is not None:
            rename_map[source] = target

    out = out.rename(columns=rename_map)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    if "股票代码" not in out.columns:
        raise RuntimeError(f"结果中缺少股票代码列，现有列: {list(out.columns)}")

    if "股票简称" not in out.columns:
        out["股票简称"] = ""
    out["基础代码"] = out["股票代码"].apply(normalize_stock_code)

    for column in [
        "竞价涨幅今日",
        "竞价换手率今日",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "开盘价:不复权今日",
        "成交金额昨日",
        "成交量昨日",
        "成交量前日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
        "个股热度排名昨日",
        "连续涨停天数昨日",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    merged = valid_frames[0].copy()
    for frame in valid_frames[1:]:
        merged = pd.merge(merged, frame, on=["股票代码", "基础代码"], how="inner", suffixes=("", "__new"))
        for column in frame.columns:
            if column in {"股票代码", "基础代码"}:
                continue
            new_column = f"{column}__new"
            if new_column not in merged.columns:
                continue
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[new_column])
            else:
                merged[column] = merged[new_column]
            merged.drop(columns=[new_column], inplace=True)

    if "股票简称" not in merged.columns:
        merged["股票简称"] = ""
    if "股票简称_x" in merged.columns:
        merged["股票简称"] = merged["股票简称_x"].combine_first(merged.get("股票简称_y"))
    return merged


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "竞价未匹配金额" not in out.columns:
        out["竞价未匹配金额"] = pd.Series(index=out.index, dtype="float64")
    if "成交金额昨日" not in out.columns:
        raise RuntimeError("缺少昨日成交金额，无法计算竞昨成交比")
    out["竞昨成交比"] = pd.to_numeric(out["竞价匹配金额_openapi"], errors="coerce") / pd.to_numeric(
        out["成交金额昨日"], errors="coerce"
    )
    out["昨日前日成交量比"] = out["成交量昨日"] / out["成交量前日"]
    out["竞价未匹配占比"] = pd.to_numeric(out["竞价未匹配金额"], errors="coerce") / pd.to_numeric(
        out["竞价匹配金额_openapi"], errors="coerce"
    )
    return out


def apply_strategy(
    df: pd.DataFrame,
    snapshot: dict[str, Any],
    trade_date: pd.Timestamp,
    sector_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    status = {
        "交易日期": trade_date.strftime("%Y-%m-%d"),
        "市场快照日期": snapshot.get("日期"),
        "市场20日高低差": snapshot.get("市场20日高低差"),
        "开仓开关": snapshot.get("开仓开关"),
        "行业强度口径": sector_source,
        "行业涨幅阈值": INDUSTRY_CHANGE_MIN,
        "前日实体阈值": PREV_BODY_MIN,
        "原始候选数": int(len(df)),
    }

    if snapshot.get("开仓开关") != "通过":
        status["金额过滤后"] = 0
        status["实体过滤后"] = 0
        status["行业过滤后"] = 0
        status["最终候选数"] = 0
        status["入选数"] = 0
        status["结果说明"] = "市场开关未通过，今日空仓"
        return df.head(0).copy(), df.head(0).copy(), status

    filtered = df.copy()
    filtered = filtered[filtered["竞价匹配金额_openapi"] >= 50_000_000].copy()
    status["金额过滤后"] = int(len(filtered))
    yesterday_body = pd.to_numeric(filtered["实体涨跌幅昨日"], errors="coerce")
    prev_body = pd.to_numeric(filtered["实体涨跌幅前日"], errors="coerce")
    filtered = filtered[(yesterday_body < prev_body) & (prev_body >= PREV_BODY_MIN)].copy()
    status["实体过滤后"] = int(len(filtered))
    if "申万一级行业涨跌幅" not in filtered.columns:
        raise RuntimeError("缺少申万一级行业涨跌幅，无法执行行业联动过滤")
    filtered = filtered[pd.to_numeric(filtered["申万一级行业涨跌幅"], errors="coerce") > INDUSTRY_CHANGE_MIN].copy()
    status["行业过滤后"] = int(len(filtered))
    filtered = filtered.sort_values(
        ["竞价未匹配占比", "竞昨成交比", "个股热度排名昨日", "基础代码"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    filtered["排序名次"] = range(1, len(filtered) + 1)
    selected = filtered.head(TOP_N).copy()
    if not selected.empty:
        selected["建议动作"] = "开盘等权买入"
        selected["建议权重"] = round(1 / len(selected), 4)
    status["最终候选数"] = int(len(filtered))
    status["入选数"] = int(len(selected))
    status["结果说明"] = "市场开关通过" if not selected.empty else "市场开关通过，但无符合条件标的"
    return filtered, selected, status


def export_outputs(
    trade_date: pd.Timestamp,
    filtered_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    status: dict[str, Any],
    queries: dict[str, str],
) -> None:
    date_text = trade_date.strftime("%Y%m%d")
    export_filtered = filtered_df.copy()
    export_selected = selected_df.copy()

    candidate_columns = [
        "排序名次",
        "股票代码",
        "股票简称",
        "申万一级行业代码",
        "申万一级行业",
        "申万一级行业涨跌幅",
        "申万一级行业涨跌幅排名",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "竞价未匹配占比",
        "竞价涨幅今日",
        "竞价换手率今日",
        "开盘价:不复权今日",
        "量比",
        "成交金额昨日",
        "成交量昨日",
        "成交量前日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
        "个股热度排名昨日",
        "连续涨停天数昨日",
        "竞昨成交比",
        "昨日前日成交量比",
    ]
    keep_candidate = [column for column in candidate_columns if column in export_filtered.columns]
    keep_selected = [column for column in candidate_columns + ["建议动作", "建议权重"] if column in export_selected.columns]

    latest_csv = OUTPUT_PREFIX.with_suffix(".csv")
    dated_csv = BASE_DIR / f"今日操作清单-{date_text}.csv"
    export_selected[keep_selected].to_csv(latest_csv, index=False, encoding="utf-8-sig")
    export_selected[keep_selected].to_csv(dated_csv, index=False, encoding="utf-8-sig")

    candidate_csv = BASE_DIR / f"今日操作候选池-{date_text}.csv"
    export_filtered[keep_candidate].to_csv(candidate_csv, index=False, encoding="utf-8-sig")

    payload = {
        "status": status,
        "queries": queries,
        "selected": export_selected[keep_selected].to_dict("records"),
        "candidates": export_filtered[keep_candidate].to_dict("records"),
    }
    latest_json = OUTPUT_PREFIX.with_suffix(".json")
    dated_json = BASE_DIR / f"今日操作清单-{date_text}.json"
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dated_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 今日操作清单 - {status['交易日期']}",
        "",
        "## 状态",
        f"- 市场快照日期: `{status.get('市场快照日期')}`",
        f"- 市场20日高低差: `{status.get('市场20日高低差')}`",
        f"- 开仓开关: `{status.get('开仓开关')}`",
        f"- 行业强度口径: `{status.get('行业强度口径')}`",
        f"- 行业涨幅阈值: `>{status.get('行业涨幅阈值')}`",
        f"- 前日实体阈值: `>={status.get('前日实体阈值')}`",
        f"- 原始候选数: `{status.get('原始候选数')}`",
        f"- 金额过滤后: `{status.get('金额过滤后', 0)}`",
        f"- 实体过滤后: `{status.get('实体过滤后', 0)}`",
        f"- 行业过滤后: `{status.get('行业过滤后', 0)}`",
        f"- 最终候选数: `{status.get('最终候选数', 0)}`",
        f"- 入选数: `{status.get('入选数', 0)}`",
        f"- 结果说明: {status.get('结果说明')}",
        "",
        "## 入选标的",
    ]
    if export_selected.empty:
        lines.append("- 今日无可操作标的")
    else:
        lines.extend(
            [
                "| 排名 | 股票代码 | 股票简称 | 一级行业 | 行业涨幅 | 行业涨幅排名 | 竞价金额 | 未匹配占比 | 竞昨成交比 | 热度排名昨日 | 建议动作 |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in export_selected.iterrows():
            lines.append(
                f"| {int(row['排序名次'])} | {row['股票代码']} | {row['股票简称']} | "
                f"{row.get('申万一级行业', '')} | {row.get('申万一级行业涨跌幅', float('nan')):.4f} | "
                f"{row.get('申万一级行业涨跌幅排名', float('nan')):.0f} | "
                f"{row['竞价匹配金额_openapi']:.0f} | {row.get('竞价未匹配占比', float('nan')):.4f} | {row['竞昨成交比']:.4f} | "
                f"{row['个股热度排名昨日']:.0f} | {row['建议动作']} |"
            )

    lines.extend(
        [
            "",
            "## 查询语句",
            f"- 基础信号: `{queries['base']}`",
            f"- 明细字段: `{queries['detail']}`",
            f"- 竞价金额: `{queries['amount']}`",
        ]
    )

    latest_md = OUTPUT_PREFIX.with_suffix(".md")
    dated_md = BASE_DIR / f"今日操作清单-{date_text}.md"
    content = "\n".join(lines)
    latest_md.write_text(content, encoding="utf-8")
    dated_md.write_text(content, encoding="utf-8")
    logging.info(
        "导出完成 trade_date=%s selected=%s latest_csv=%s dated_csv=%s candidate_csv=%s",
        date_text,
        len(export_selected),
        latest_csv,
        dated_csv,
        candidate_csv,
    )


def format_push_number(value: Any, digits: int = 4) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "-"
    return f"{float(number):.{digits}f}"


def build_pushplus_content(trade_date: pd.Timestamp, selected_df: pd.DataFrame, status: dict[str, Any]) -> str:
    lines = [
        f"交易日期: {trade_date.strftime('%Y-%m-%d')}",
        f"市场开关: {status.get('开仓开关')} / 市场20日高低差={status.get('市场20日高低差')}",
        f"行业口径: {status.get('行业强度口径')}",
        f"过滤: 原始{status.get('原始候选数', 0)} -> 金额{status.get('金额过滤后', 0)} -> 实体{status.get('实体过滤后', 0)} -> 行业{status.get('行业过滤后', 0)} -> 入选{status.get('入选数', 0)}",
        f"说明: {status.get('结果说明')}",
        "",
    ]

    if selected_df.empty:
        lines.append("今日无可操作标的")
        return "\n".join(lines)

    lines.append(f"今日前{TOP_N}标的:")
    for _, row in selected_df.iterrows():
        lines.extend(
            [
                f"{int(row['排序名次'])}. {row['股票简称']}（{row['股票代码']}）",
                f"行业: {row.get('申万一级行业', '-')}",
                f"建议权重: {format_push_number(row.get('建议权重'), 4)}",
                f"竞价金额: {format_push_number(row.get('竞价匹配金额_openapi'), 0)}",
                f"未匹配占比: {format_push_number(row.get('竞价未匹配占比'), 4)}",
                f"竞昨成交比: {format_push_number(row.get('竞昨成交比'), 4)}",
                f"昨日热度排名: {format_push_number(row.get('个股热度排名昨日'), 0)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def send_pushplus(title: str, content: str) -> None:
    payload = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "txt",
    }
    response = requests.post(PUSHPLUS_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
    response.raise_for_status()
    logging.info("PushPlus推送成功 title=%s response=%s", title, response.text)
    print(f"PushPlus推送结果: {response.text}")


def main() -> None:
    args = parse_args()
    logging.info("开始运行 args=%s cwd=%s python=%s", vars(args), Path.cwd(), sys.executable)
    explicit_trade_date = (
        pd.Timestamp(args.trade_date).normalize()
        if args.trade_date
        else None
    )
    today_now = pd.Timestamp(date.today()).normalize()
    historical_replay = explicit_trade_date is not None and explicit_trade_date < today_now
    if explicit_trade_date is None:
        now = datetime.now()
        if now.time() < datetime.strptime("09:25", "%H:%M").time():
            raise RuntimeError("当前时间早于 09:25，今日竞价数据尚不可用。请 09:25 后再运行，或使用 --trade-date 做历史回放。")

    today_ts, prev_ts, prev2_ts = pick_trade_dates(explicit_trade_date.date() if explicit_trade_date is not None else None)
    logging.info(
        "交易日定位 today=%s prev=%s prev2=%s historical_replay=%s",
        today_ts.strftime("%Y-%m-%d"),
        prev_ts.strftime("%Y-%m-%d"),
        prev2_ts.strftime("%Y-%m-%d"),
        historical_replay,
    )
    required_snapshot_date = None if historical_replay else prev_ts
    snapshot = read_market_snapshot(today_ts, historical_replay, required_snapshot_date)
    logging.info(
        "市场快照 date=%s diff20=%s switch=%s",
        snapshot.get("日期"),
        snapshot.get("市场20日高低差"),
        snapshot.get("开仓开关"),
    )
    cookies = resolve_cookies()
    queries = build_queries(today_ts, prev_ts, prev2_ts)
    date_map = {
        date_token(today_ts): "今日",
        date_token(prev_ts): "昨日",
        date_token(prev2_ts): "前日",
    }

    frames = []
    for label in ["base", "detail", "amount"]:
        frame = query_wencai(queries[label], cookies)
        logging.info("问财查询完成 label=%s rows=%s", label, len(frame))
        standardized = standardize_frame(frame, date_map)
        frames.append(standardized)

    merged = merge_frames(frames)
    if merged.empty:
        raise RuntimeError("问财返回为空，无法生成操作清单")

    merged = compute_factors(merged)
    merged, sector_source = attach_sector_context(merged, today_ts, historical_replay)
    filtered, selected, status = apply_strategy(merged, snapshot, today_ts, sector_source)
    export_outputs(today_ts, filtered, selected, status, queries)
    should_push = not args.no_push and (not historical_replay or args.push)
    logging.info("推送判断 should_push=%s no_push=%s push=%s", should_push, args.no_push, args.push)
    if should_push:
        push_title = f"{today_ts.strftime('%Y-%m-%d')} 竞价爬升操作清单"
        push_content = build_pushplus_content(today_ts, selected, status)
        send_pushplus(push_title, push_content)
    logging.info(
        "运行成功 trade_date=%s raw=%s amount=%s body=%s industry=%s selected=%s",
        status.get("交易日期"),
        status.get("原始候选数"),
        status.get("金额过滤后"),
        status.get("实体过滤后"),
        status.get("行业过滤后"),
        status.get("入选数"),
    )

    print(f"交易日期: {status['交易日期']}")
    print(f"市场开关: {status['开仓开关']} / 市场20日高低差={status['市场20日高低差']}")
    print(f"行业口径: {status['行业强度口径']} / 阈值=涨幅>{status['行业涨幅阈值']}")
    print(f"原始候选数: {status['原始候选数']}")
    print(f"金额过滤后: {status['金额过滤后']}")
    print(f"实体过滤后: {status['实体过滤后']}")
    print(f"行业过滤后: {status['行业过滤后']}")
    print(f"最终候选数: {status['最终候选数']}")
    print(f"入选数: {status['入选数']}")
    if selected.empty:
        print("今日无可操作标的")
    else:
        print(f"今日前{TOP_N}标的:")
        print(
            selected[
                [
                    "排序名次",
                    "股票代码",
                    "股票简称",
                    "申万一级行业",
                    "申万一级行业涨跌幅",
                    "申万一级行业涨跌幅排名",
                    "竞价匹配金额_openapi",
                    "竞价未匹配占比",
                    "竞昨成交比",
                    "个股热度排名昨日",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("生成失败")
        print(f"生成失败: {exc}")
        raise SystemExit(1)
