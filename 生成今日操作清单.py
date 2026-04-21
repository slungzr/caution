from __future__ import annotations

import json
import os
import re
import time
import argparse
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import pywencai

from wencai_direct import fetch_query_dataframe, load_cookie as load_legacy_cookie


BASE_DIR = Path(__file__).resolve().parent
TRADE_CALENDAR_CSV = BASE_DIR / "trade_calendar.csv"
COOKIE_FILE = BASE_DIR / "wencai_cookie.txt"
PYWENCAI_SCRIPT = BASE_DIR / "竞价因子补充_pywencai.py"
UNIFIEDWAP_SCRIPT = BASE_DIR / "wencai_unifiedwap.py"
MARKET_SNAPSHOT_JSON = BASE_DIR / "最新市场宽度.json"
OUTPUT_PREFIX = BASE_DIR / "今日操作清单"
TOP_N = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9:25 后生成今日操作清单")
    parser.add_argument(
        "--trade-date",
        help="指定交易日期，格式 YYYYMMDD，用于历史回放或盘后验证",
    )
    return parser.parse_args()


def read_snapshot(snapshot_path: Path) -> dict[str, Any]:
    if not snapshot_path.exists():
        raise FileNotFoundError(f"未找到市场开关快照: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


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
        f"{today_cn}竞价金额，{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
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
        "开盘价:不复权今日": ["开盘价:不复权今日", "开盘价今日", "开盘价:不复权"],
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
        "开盘价:不复权今日",
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
    out["昨收估算"] = out["开盘价:不复权今日"] / (1 + out["竞价涨幅今日"] / 100)
    out["昨日成交额估算"] = out["昨收估算"] * out["成交量昨日"]
    out["竞昨成交比估算"] = out["竞价匹配金额_openapi"] / out["昨日成交额估算"]
    out["昨日前日成交量比"] = out["成交量昨日"] / out["成交量前日"]
    return out


def apply_strategy(df: pd.DataFrame, snapshot: dict[str, Any], trade_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    status = {
        "交易日期": trade_date.strftime("%Y-%m-%d"),
        "市场快照日期": snapshot.get("日期"),
        "市场20日高低差": snapshot.get("市场20日高低差"),
        "开仓开关": snapshot.get("开仓开关"),
        "原始候选数": int(len(df)),
    }

    if snapshot.get("开仓开关") != "通过":
        status["金额过滤后"] = 0
        status["实体过滤后"] = 0
        status["最终候选数"] = 0
        status["结果说明"] = "市场开关未通过，今日空仓"
        return df.head(0).copy(), df.head(0).copy(), status

    filtered = df.copy()
    filtered = filtered[filtered["竞价匹配金额_openapi"] >= 50_000_000].copy()
    status["金额过滤后"] = int(len(filtered))
    filtered = filtered[filtered["实体涨跌幅昨日"] < filtered["实体涨跌幅前日"]].copy()
    status["实体过滤后"] = int(len(filtered))
    filtered = filtered.sort_values(
        ["竞昨成交比估算", "个股热度排名昨日", "基础代码"],
        ascending=[False, True, True],
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
        "竞价匹配金额_openapi",
        "竞价涨幅今日",
        "竞价换手率今日",
        "开盘价:不复权今日",
        "成交量昨日",
        "成交量前日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
        "个股热度排名昨日",
        "连续涨停天数昨日",
        "昨收估算",
        "昨日成交额估算",
        "竞昨成交比估算",
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
        f"- 原始候选数: `{status.get('原始候选数')}`",
        f"- 金额过滤后: `{status.get('金额过滤后', 0)}`",
        f"- 实体过滤后: `{status.get('实体过滤后', 0)}`",
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
                "| 排名 | 股票代码 | 股票简称 | 竞价金额 | 竞昨成交比 | 热度排名昨日 | 建议动作 |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in export_selected.iterrows():
            lines.append(
                f"| {int(row['排序名次'])} | {row['股票代码']} | {row['股票简称']} | "
                f"{row['竞价匹配金额_openapi']:.0f} | {row['竞昨成交比估算']:.4f} | "
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


def main() -> None:
    args = parse_args()
    explicit_trade_date = (
        pd.Timestamp(args.trade_date).normalize()
        if args.trade_date
        else None
    )
    if explicit_trade_date is None:
        now = datetime.now()
        if now.time() < datetime.strptime("09:25", "%H:%M").time():
            raise RuntimeError("当前时间早于 09:25，今日竞价数据尚不可用。请 09:25 后再运行，或使用 --trade-date 做历史回放。")

    today_ts, prev_ts, prev2_ts = pick_trade_dates(explicit_trade_date.date() if explicit_trade_date is not None else None)
    snapshot = read_snapshot(MARKET_SNAPSHOT_JSON)
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
        standardized = standardize_frame(frame, date_map)
        frames.append(standardized)

    merged = merge_frames(frames)
    if merged.empty:
        raise RuntimeError("问财返回为空，无法生成操作清单")

    merged = compute_factors(merged)
    filtered, selected, status = apply_strategy(merged, snapshot, today_ts)
    export_outputs(today_ts, filtered, selected, status, queries)

    print(f"交易日期: {status['交易日期']}")
    print(f"市场开关: {status['开仓开关']} / 市场20日高低差={status['市场20日高低差']}")
    print(f"原始候选数: {status['原始候选数']}")
    print(f"金额过滤后: {status['金额过滤后']}")
    print(f"实体过滤后: {status['实体过滤后']}")
    print(f"最终候选数: {status['最终候选数']}")
    print(f"入选数: {status['入选数']}")
    if selected.empty:
        print("今日无可操作标的")
    else:
        print("今日前3标的:")
        print(
            selected[
                [
                    "排序名次",
                    "股票代码",
                    "股票简称",
                    "竞价匹配金额_openapi",
                    "竞昨成交比估算",
                    "个股热度排名昨日",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"生成失败: {exc}")
        raise SystemExit(1)
