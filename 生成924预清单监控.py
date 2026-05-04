from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DAILY_SCRIPT = BASE_DIR / "生成今日操作清单.py"
OUTPUT_PREFIX = BASE_DIR / "924预清单"
MONITOR_LOG_PREFIX = BASE_DIR / "924预清单监控记录"
DEFAULT_START_TIME = "09:20:30"
DEFAULT_END_TIME = "09:24:50"
DEFAULT_INTERVAL_SECONDS = 15
DEFAULT_ENTRY_BUFFER_PCT = 0.005


def load_daily_module():
    spec = importlib.util.spec_from_file_location("daily_operation_list_for_924", DAILY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载正式清单脚本: {DAILY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


daily = load_daily_module()


@dataclass
class PreopenRunResult:
    trade_date: pd.Timestamp
    filtered: pd.DataFrame
    selected: pd.DataFrame
    status: dict[str, Any]
    queries: dict[str, str]
    output_paths: dict[str, Path]
    signature: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是大于等于 1 的整数")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须大于等于 0")
    return parsed


def parse_clock(value: str) -> dt_time:
    try:
        return datetime.strptime(value, "%H:%M:%S").time()
    except ValueError:
        try:
            return datetime.strptime(value, "%H:%M").time()
        except ValueError as exc:
            raise argparse.ArgumentTypeError("时间格式必须是 HH:MM 或 HH:MM:SS") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9:24 预清单/监控：提前确认集合竞价可执行候选")
    parser.add_argument("--trade-date", help="指定交易日期，格式 YYYYMMDD，用于历史回放或手工校验")
    parser.add_argument("--once", action="store_true", help="只刷新一次，不进入 9:20-9:24 监控循环")
    parser.add_argument("--start-time", default=DEFAULT_START_TIME, type=parse_clock, help="监控开始时间，默认 09:20:30")
    parser.add_argument("--end-time", default=DEFAULT_END_TIME, type=parse_clock, help="监控截止时间，默认 09:24:50")
    parser.add_argument(
        "--interval-seconds",
        type=positive_int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="刷新间隔秒数，默认 15",
    )
    parser.add_argument("--push", action="store_true", help="监控结束后推送最终 9:24 预清单")
    parser.add_argument("--push-each", action="store_true", help="每次刷新都推送；一般不建议，容易打扰")
    parser.add_argument("--no-push", action="store_true", help="本次运行不发送 PushPlus 推送")
    parser.add_argument(
        "--allow-incomplete-auction-bid",
        action="store_true",
        help="竞昨成交比可用但未匹配占比缺失时，仍允许输出集合竞价买入建议；默认只作为观察名单",
    )
    parser.add_argument(
        "--entry-buffer-pct",
        type=non_negative_float,
        default=DEFAULT_ENTRY_BUFFER_PCT,
        help="建议限价相对预估匹配价的上浮比例，默认 0.005，即 0.5%%",
    )
    parser.add_argument(
        "--top-n",
        type=positive_int,
        default=daily.TOP_N,
        help="配置最大入选标的数量，默认 3；动态持仓开启时按市场强弱自动降到 2 或 1",
    )
    parser.add_argument(
        "--fixed-top-n",
        action="store_true",
        help="关闭动态持仓，严格使用 --top-n 指定的最大入选数",
    )
    parser.add_argument(
        "--min-auction-ratio",
        type=float,
        default=daily.MIN_AUCTION_TO_YESTERDAY_RATIO,
        help="竞昨成交比最低阈值，默认 0.022",
    )
    parser.add_argument(
        "--no-auction-ratio-filter",
        action="store_true",
        help="关闭竞昨成交比过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--min-unmatched-ratio",
        type=float,
        default=daily.DEFAULT_MIN_UNMATCHED_RATIO,
        help="竞价未匹配占比最低阈值，默认不启用",
    )
    industry_group = parser.add_mutually_exclusive_group()
    industry_group.add_argument(
        "--industry-filter",
        dest="industry_filter",
        action="store_true",
        default=daily.DEFAULT_INDUSTRY_FILTER_ENABLED,
        help="启用申万一级行业涨幅>0过滤",
    )
    industry_group.add_argument(
        "--no-industry-filter",
        dest="industry_filter",
        action="store_false",
        help="关闭申万一级行业涨幅过滤，当前默认",
    )
    return parser.parse_args()


def build_preopen_queries(today_ts: pd.Timestamp, prev_ts: pd.Timestamp, prev2_ts: pd.Timestamp) -> dict[str, str]:
    today_cn = daily.cn_date(today_ts)
    prev_cn = daily.cn_date(prev_ts)
    prev2_cn = daily.cn_date(prev2_ts)

    base_query = (
        f"{today_cn}9点24分最低价>{today_cn}9点23分最高价，"
        f"{today_cn}9点23分最低价>={today_cn}9点22分最高价，"
        f"{today_cn}9点22分最低价>={today_cn}9点21分最高价，"
        f"{today_cn}9点21分最低价>={today_cn}9点20分最高价，"
        f"{today_cn}竞价涨幅，{today_cn}竞价换手率，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    detail_query = (
        f"{prev_cn}成交量，{prev2_cn}成交量，"
        f"{prev_cn}实体涨跌幅，{prev2_cn}实体涨跌幅，"
        f"{prev_cn}个股热度排名，{prev_cn}连续涨停天数，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    amount_query = (
        f"{today_cn}竞价匹配价，{today_cn}竞价金额，{today_cn}竞价未匹配金额，"
        f"{today_cn}竞价量，{today_cn}竞价未匹配量，{prev_cn}成交金额，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    return {"base": base_query, "detail": detail_query, "amount": amount_query}


def explicit_trade_date(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value).normalize() if value else None


def safe_attach_sector_context(
    merged: pd.DataFrame,
    trade_date: pd.Timestamp,
    historical_replay: bool,
    industry_filter_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        return daily.attach_sector_context(merged, trade_date, historical_replay)
    except Exception as exc:
        if industry_filter_enabled:
            raise
        logging.warning("9:24预清单行业上下文获取失败，默认降级继续: %s", exc)
        out = merged.copy()
        for column in ["申万一级行业代码", "申万一级行业", "申万一级行业涨跌幅", "申万一级行业涨跌幅排名"]:
            if column not in out.columns:
                out[column] = pd.NA
        source = f"行业实时获取失败，预清单降级继续: {type(exc).__name__}"
        return out, pd.DataFrame(), source


def compute_preopen_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in [
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "成交金额昨日",
        "成交量昨日",
        "成交量前日",
    ]:
        if column not in out.columns:
            out[column] = pd.Series(float("nan"), index=out.index, dtype="float64")

    auction_amount = pd.to_numeric(out["竞价匹配金额_openapi"], errors="coerce")
    unmatched_amount = pd.to_numeric(out["竞价未匹配金额"], errors="coerce")
    yesterday_amount = pd.to_numeric(out["成交金额昨日"], errors="coerce")
    yesterday_volume = pd.to_numeric(out["成交量昨日"], errors="coerce")
    prev_volume = pd.to_numeric(out["成交量前日"], errors="coerce")

    out["竞昨成交比"] = auction_amount / yesterday_amount
    out["竞价未匹配占比"] = unmatched_amount / auction_amount
    out["昨日前日成交量比"] = yesterday_volume / prev_volume
    return out


def has_numeric_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()


def assess_preopen_data_readiness(df: pd.DataFrame, allow_incomplete_bid: bool = False) -> dict[str, Any]:
    auction_amount_ready = has_numeric_data(df, "竞价匹配金额_openapi")
    yesterday_amount_ready = has_numeric_data(df, "成交金额昨日")
    unmatched_ready = has_numeric_data(df, "竞价未匹配金额") and has_numeric_data(df, "竞价未匹配占比")
    ratio_ready = auction_amount_ready and yesterday_amount_ready and has_numeric_data(df, "竞昨成交比")

    if ratio_ready and unmatched_ready:
        level = "完整"
        can_bid = True
        note = "竞价金额、竞昨成交比、未匹配占比均可用，可按预委托清单执行。"
    elif ratio_ready:
        level = "半完整"
        can_bid = bool(allow_incomplete_bid)
        note = (
            "竞价金额/竞昨成交比可用，但未匹配占比缺失；排序不能完全复刻主策略，"
            "默认只建议观察，除非显式允许不完整数据委托。"
        )
    else:
        level = "观察"
        can_bid = False
        note = "竞价金额或竞昨成交比不可用，当前只能生成预观察名单，不能作为集合竞价委托依据。"

    return {
        "预清单数据成熟度": level,
        "竞价金额可用": auction_amount_ready,
        "昨日成交额可用": yesterday_amount_ready,
        "竞昨成交比可用": ratio_ready,
        "未匹配占比可用": unmatched_ready,
        "允许不完整数据委托": allow_incomplete_bid,
        "可集合竞价委托": can_bid,
        "数据成熟度说明": note,
    }


def apply_preopen_observation_strategy(
    df: pd.DataFrame,
    snapshot: dict[str, Any],
    trade_date: pd.Timestamp,
    sector_source: str,
    top_n: int,
    industry_filter_enabled: bool,
    min_unmatched_ratio: float | None,
    dynamic_top_n_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    effective_top_n = daily.resolve_effective_top_n(snapshot, top_n, dynamic_top_n_enabled)
    status = {
        "交易日期": trade_date.strftime("%Y-%m-%d"),
        "市场快照日期": snapshot.get("日期"),
        "市场20日高低差": snapshot.get("市场20日高低差"),
        "开仓开关": snapshot.get("开仓开关"),
        "行业强度口径": sector_source,
        "行业过滤启用": industry_filter_enabled,
        "行业涨幅阈值": daily.INDUSTRY_CHANGE_MIN,
        "前日实体阈值": daily.PREV_BODY_MIN,
        "竞昨成交比阈值": None,
        "未匹配占比阈值": min_unmatched_ratio,
        "配置最大入选数": top_n,
        "最大入选数": effective_top_n,
        "动态持仓启用": dynamic_top_n_enabled,
        "动态持仓强市场阈值": daily.DYNAMIC_TOP_N_STRONG_MARKET_DIFF,
        "动态持仓中市场阈值": daily.DYNAMIC_TOP_N_MIDDLE_MARKET_DIFF,
        "盘后弱承接风控阈值": daily.ENTRY_DAY_LOW_FROM_OPEN_RISK_EXIT,
        "原始候选数": int(len(df)),
        "金额过滤后": int(len(df)),
        "竞昨过滤后": int(len(df)),
    }

    if snapshot.get("开仓开关") != "通过":
        status.update(
            {
                "实体过滤后": 0,
                "行业过滤后": 0,
                "未匹配过滤后": 0,
                "最终候选数": 0,
                "入选数": 0,
                "结果说明": "市场开关未通过，今日空仓",
            }
        )
        return df.head(0).copy(), df.head(0).copy(), status

    filtered = df.copy()
    yesterday_body = pd.to_numeric(filtered["实体涨跌幅昨日"], errors="coerce")
    prev_body = pd.to_numeric(filtered["实体涨跌幅前日"], errors="coerce")
    filtered = filtered[(yesterday_body < prev_body) & (prev_body >= daily.PREV_BODY_MIN)].copy()
    status["实体过滤后"] = int(len(filtered))

    if industry_filter_enabled:
        if "申万一级行业涨跌幅" not in filtered.columns:
            raise RuntimeError("缺少申万一级行业涨跌幅，无法执行行业联动过滤")
        filtered = filtered[
            pd.to_numeric(filtered["申万一级行业涨跌幅"], errors="coerce") > daily.INDUSTRY_CHANGE_MIN
        ].copy()
    status["行业过滤后"] = int(len(filtered))

    if min_unmatched_ratio is not None and has_numeric_data(filtered, "竞价未匹配占比"):
        unmatched_ratio = pd.to_numeric(filtered["竞价未匹配占比"], errors="coerce")
        filtered = filtered[unmatched_ratio >= min_unmatched_ratio].copy()
    status["未匹配过滤后"] = int(len(filtered))

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in [
        ("个股热度排名昨日", True),
        ("竞价涨幅今日", False),
        ("竞价换手率今日", False),
        ("基础代码", True),
    ]:
        if column in filtered.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)
    if sort_columns:
        filtered = filtered.sort_values(sort_columns, ascending=ascending, kind="stable").reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)
    filtered["排序名次"] = range(1, len(filtered) + 1)

    selected = filtered.head(effective_top_n).copy()
    if not selected.empty:
        selected["建议权重"] = round(1 / len(selected), 4)

    status["最终候选数"] = int(len(filtered))
    status["入选数"] = int(len(selected))
    status["结果说明"] = "预观察模式：竞价关键成交字段不完整，等待9:25正式清单确认"
    return filtered, selected, status


def decorate_execution_columns(
    selected: pd.DataFrame,
    entry_buffer_pct: float,
    can_bid: bool = True,
    readiness_note: str | None = None,
) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out

    if can_bid:
        out["建议动作"] = "9:24:50前确认后集合竞价买入"
    else:
        out["建议动作"] = "仅预观察，等待9:25正式清单确认"

    if "竞价匹配价今日" in out.columns:
        match_price = pd.to_numeric(out["竞价匹配价今日"], errors="coerce")
    else:
        match_price = pd.Series(float("nan"), index=out.index, dtype="float64")

    if can_bid:
        out["建议限价上限"] = (match_price * (1 + entry_buffer_pct)).round(2)
        out.loc[match_price.isna() | (match_price <= 0), "建议限价上限"] = pd.NA
        out["执行备注"] = (
            "限价上限用于提高集合竞价成交概率；集合竞价同一成交价撮合，"
            "不代表一定按上限成交。若9:25最终清单变化或最终匹配价明显跳升，以正式清单/纪律为准。"
        )
    else:
        out["建议限价上限"] = pd.NA
        out["执行备注"] = readiness_note or "竞价关键字段不完整，当前只做观察，不建议直接集合竞价委托。"
    return out


def selected_signature(selected: pd.DataFrame) -> str:
    if selected.empty or "股票代码" not in selected.columns:
        return "EMPTY"
    codes = selected["股票代码"].fillna("").astype(str).tolist()
    return "|".join(codes)


def export_preopen_outputs(
    trade_date: pd.Timestamp,
    filtered_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    status: dict[str, Any],
    queries: dict[str, str],
) -> dict[str, Path]:
    date_text = trade_date.strftime("%Y%m%d")
    candidate_columns = [
        "排序名次",
        "股票代码",
        "股票简称",
        "申万一级行业代码",
        "申万一级行业",
        "申万一级行业涨跌幅",
        "申万一级行业涨跌幅排名",
        "竞价匹配价今日",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "竞价未匹配占比",
        "竞价量今日",
        "竞价未匹配量今日",
        "竞价涨幅今日",
        "竞价换手率今日",
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
    selected_columns = candidate_columns + ["建议动作", "建议权重", "建议限价上限", "执行备注"]
    keep_candidate = [column for column in candidate_columns if column in filtered_df.columns]
    keep_selected = [column for column in selected_columns if column in selected_df.columns]

    latest_csv = OUTPUT_PREFIX.with_suffix(".csv")
    dated_csv = BASE_DIR / f"924预清单-{date_text}.csv"
    candidate_csv = BASE_DIR / f"924预候选池-{date_text}.csv"
    latest_json = OUTPUT_PREFIX.with_suffix(".json")
    dated_json = BASE_DIR / f"924预清单-{date_text}.json"
    latest_md = OUTPUT_PREFIX.with_suffix(".md")
    dated_md = BASE_DIR / f"924预清单-{date_text}.md"

    selected_df[keep_selected].to_csv(latest_csv, index=False, encoding="utf-8-sig")
    selected_df[keep_selected].to_csv(dated_csv, index=False, encoding="utf-8-sig")
    filtered_df[keep_candidate].to_csv(candidate_csv, index=False, encoding="utf-8-sig")

    payload = {
        "status": status,
        "queries": queries,
        "selected": selected_df[keep_selected].to_dict("records"),
        "candidates": filtered_df[keep_candidate].to_dict("records"),
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, default=daily.to_json_safe)
    latest_json.write_text(json_text, encoding="utf-8")
    dated_json.write_text(json_text, encoding="utf-8")

    lines = [
        f"# 9:24预清单 - {status['交易日期']}",
        "",
        "## 执行说明",
        "- 这是 9:24 预清单，不是 9:25 最终清单。",
        "- 只有数据成熟度为“完整”时，才建议按预清单准备集合竞价委托。",
        "- 若数据成熟度为“半完整/观察”，只适合提前观察或录入候选，最终等 9:25 正式清单确认。",
        "- 如果 9:25 正式清单变化、最终匹配价明显跳升，或者市场开关不通过，以正式清单和交易纪律为准。",
        f"- 建议限价上限: 预估匹配价上浮 `{status.get('建议限价上浮比例')}`。",
        "",
        "## 状态",
        f"- 采集时间: `{status.get('采集时间')}`",
        f"- 采集轮次: `{status.get('采集轮次')}`",
        f"- 数据成熟度: `{status.get('预清单数据成熟度')}`",
        f"- 可集合竞价委托: `{status.get('可集合竞价委托')}`",
        f"- 竞价金额可用: `{status.get('竞价金额可用')}`",
        f"- 竞昨成交比可用: `{status.get('竞昨成交比可用')}`",
        f"- 未匹配占比可用: `{status.get('未匹配占比可用')}`",
        f"- 数据说明: {status.get('数据成熟度说明')}",
        f"- 市场快照日期: `{status.get('市场快照日期')}`",
        f"- 市场20日高低差: `{status.get('市场20日高低差')}`",
        f"- 开仓开关: `{status.get('开仓开关')}`",
        f"- 行业强度口径: `{status.get('行业强度口径')}`",
        f"- 行业过滤: `{'启用' if status.get('行业过滤启用') else '关闭'}`",
        f"- 竞昨成交比阈值: `{daily.format_ratio_threshold(status.get('竞昨成交比阈值'))}`",
        f"- 未匹配占比阈值: `{daily.format_ratio_threshold(status.get('未匹配占比阈值'))}`",
        f"- 动态持仓: `{'启用' if status.get('动态持仓启用') else '关闭'}`",
        f"- 配置TOP{status.get('配置最大入选数')} -> 今日TOP{status.get('最大入选数')}",
        f"- 过滤: 原始{status.get('原始候选数', 0)} -> 金额{status.get('金额过滤后', 0)} -> 竞昨{status.get('竞昨过滤后', 0)} -> 实体{status.get('实体过滤后', 0)} -> 行业{status.get('行业过滤后', 0)} -> 未匹配{status.get('未匹配过滤后', 0)} -> 入选{status.get('入选数', 0)}",
        f"- 结果说明: {status.get('结果说明')}",
        "",
        "## 入选标的",
    ]
    if selected_df.empty:
        lines.append("- 当前无预入选标的")
    else:
        lines.extend(
            [
                "| 排名 | 股票代码 | 股票简称 | 竞价匹配价 | 建议限价上限 | 竞价金额 | 未匹配占比 | 竞昨成交比 | 热度排名昨日 | 建议动作 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in selected_df.iterrows():
            lines.append(
                f"| {int(row['排序名次'])} | {row['股票代码']} | {row['股票简称']} | "
                f"{daily.format_push_number(row.get('竞价匹配价今日'), 2)} | "
                f"{daily.format_push_number(row.get('建议限价上限'), 2)} | "
                f"{daily.format_push_number(row.get('竞价匹配金额_openapi'), 0)} | "
                f"{daily.format_push_number(row.get('竞价未匹配占比'), 4)} | "
                f"{daily.format_push_number(row.get('竞昨成交比'), 4)} | "
                f"{daily.format_push_number(row.get('个股热度排名昨日'), 0)} | "
                f"{row.get('建议动作', '')} |"
            )

    lines.extend(
        [
            "",
            "## 查询语句",
            f"- 9:24基础信号: `{queries['base']}`",
            f"- 昨日前日明细: `{queries['detail']}`",
            f"- 竞价金额/价格: `{queries['amount']}`",
        ]
    )
    md_text = "\n".join(lines)
    latest_md.write_text(md_text, encoding="utf-8")
    dated_md.write_text(md_text, encoding="utf-8")

    return {
        "latest_csv": latest_csv,
        "dated_csv": dated_csv,
        "candidate_csv": candidate_csv,
        "latest_json": latest_json,
        "dated_json": dated_json,
        "latest_md": latest_md,
        "dated_md": dated_md,
    }


def build_preopen_push_content(result: PreopenRunResult) -> str:
    status = result.status
    selected_df = result.selected
    lines = [
        f"9:24预清单 / {result.trade_date.strftime('%Y-%m-%d')}",
        f"采集时间: {status.get('采集时间')} / 第{status.get('采集轮次')}轮",
        f"数据成熟度: {status.get('预清单数据成熟度')} / 可委托={status.get('可集合竞价委托')}",
        f"数据说明: {status.get('数据成熟度说明')}",
        f"市场开关: {status.get('开仓开关')} / 市场20日高低差={status.get('市场20日高低差')}",
        f"动态持仓: 配置TOP{status.get('配置最大入选数')} -> 今日TOP{status.get('最大入选数')}",
        f"过滤: 原始{status.get('原始候选数', 0)} -> 金额{status.get('金额过滤后', 0)} -> 竞昨{status.get('竞昨过滤后', 0)} -> 实体{status.get('实体过滤后', 0)} -> 行业{status.get('行业过滤后', 0)} -> 未匹配{status.get('未匹配过滤后', 0)} -> 入选{status.get('入选数', 0)}",
        "执行: 只有可委托=True时才考虑9:24:45-9:24:55集合竞价；否则等9:25正式清单。",
        "",
    ]
    if selected_df.empty:
        lines.append("当前无预入选标的")
        return "\n".join(lines)

    for _, row in selected_df.iterrows():
        lines.extend(
            [
                f"{int(row['排序名次'])}. {row['股票简称']}（{row['股票代码']}）",
                f"预估匹配价: {daily.format_push_number(row.get('竞价匹配价今日'), 2)}",
                f"建议限价上限: {daily.format_push_number(row.get('建议限价上限'), 2)}",
                f"建议权重: {daily.format_push_number(row.get('建议权重'), 4)}",
                f"竞价金额: {daily.format_push_number(row.get('竞价匹配金额_openapi'), 0)}",
                f"未匹配占比: {daily.format_push_number(row.get('竞价未匹配占比'), 4)}",
                f"竞昨成交比: {daily.format_push_number(row.get('竞昨成交比'), 4)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def append_monitor_log(result: PreopenRunResult) -> None:
    date_text = result.trade_date.strftime("%Y%m%d")
    log_path = BASE_DIR / f"{MONITOR_LOG_PREFIX.name}-{date_text}.csv"
    rows: list[dict[str, Any]] = []
    if result.selected.empty:
        rows.append(
            {
                "交易日期": result.trade_date.strftime("%Y-%m-%d"),
                "采集时间": result.status.get("采集时间"),
                "采集轮次": result.status.get("采集轮次"),
                "签名": result.signature,
                "入选数": 0,
                "预清单数据成熟度": result.status.get("预清单数据成熟度"),
                "可集合竞价委托": result.status.get("可集合竞价委托"),
                "结果说明": result.status.get("结果说明"),
            }
        )
    else:
        for _, row in result.selected.iterrows():
            rows.append(
                {
                    "交易日期": result.trade_date.strftime("%Y-%m-%d"),
                    "采集时间": result.status.get("采集时间"),
                    "采集轮次": result.status.get("采集轮次"),
                    "签名": result.signature,
                    "入选数": result.status.get("入选数"),
                    "预清单数据成熟度": result.status.get("预清单数据成熟度"),
                    "可集合竞价委托": result.status.get("可集合竞价委托"),
                    "排序名次": row.get("排序名次"),
                    "股票代码": row.get("股票代码"),
                    "股票简称": row.get("股票简称"),
                    "竞价匹配价今日": row.get("竞价匹配价今日"),
                    "建议限价上限": row.get("建议限价上限"),
                    "竞价匹配金额_openapi": row.get("竞价匹配金额_openapi"),
                    "竞价未匹配占比": row.get("竞价未匹配占比"),
                    "竞昨成交比": row.get("竞昨成交比"),
                    "结果说明": result.status.get("结果说明"),
                }
            )
    frame = pd.DataFrame(rows)
    if log_path.exists():
        frame.to_csv(log_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        frame.to_csv(log_path, index=False, encoding="utf-8-sig")


def run_preopen_once(args: argparse.Namespace, iteration: int) -> PreopenRunResult:
    explicit_date = explicit_trade_date(args.trade_date)
    today_now = pd.Timestamp(date.today()).normalize()
    historical_replay = explicit_date is not None and explicit_date < today_now
    today_ts, prev_ts, prev2_ts = daily.pick_trade_dates(explicit_date.date() if explicit_date is not None else None)
    required_snapshot_date = None if historical_replay else prev_ts
    snapshot = daily.read_market_snapshot(today_ts, historical_replay, required_snapshot_date)

    cookies = daily.resolve_cookies()
    queries = build_preopen_queries(today_ts, prev_ts, prev2_ts)
    date_map = {
        daily.date_token(today_ts): "今日",
        daily.date_token(prev_ts): "昨日",
        daily.date_token(prev2_ts): "前日",
    }

    frames: list[pd.DataFrame] = []
    for label in ["base", "detail", "amount"]:
        frame = daily.query_wencai(queries[label], cookies)
        logging.info("9:24预清单问财查询完成 label=%s rows=%s", label, len(frame))
        frames.append(daily.standardize_frame(frame, date_map))

    merged = daily.merge_frames(frames)
    if merged.empty:
        raise RuntimeError("问财返回为空，无法生成 9:24 预清单")
    merged = compute_preopen_factors(merged)
    readiness = assess_preopen_data_readiness(merged, args.allow_incomplete_auction_bid)
    merged, _sector_df, sector_source = safe_attach_sector_context(
        merged,
        today_ts,
        historical_replay,
        args.industry_filter,
    )
    min_auction_ratio = None if args.no_auction_ratio_filter else args.min_auction_ratio
    if readiness["竞昨成交比可用"]:
        filtered, selected, status = daily.apply_strategy(
            merged,
            snapshot,
            today_ts,
            sector_source,
            min_auction_ratio,
            args.top_n,
            args.industry_filter,
            args.min_unmatched_ratio,
            not args.fixed_top_n,
        )
    else:
        filtered, selected, status = apply_preopen_observation_strategy(
            merged,
            snapshot,
            today_ts,
            sector_source,
            args.top_n,
            args.industry_filter,
            args.min_unmatched_ratio,
            not args.fixed_top_n,
        )

    status.update(readiness)
    selected = decorate_execution_columns(
        selected,
        args.entry_buffer_pct,
        bool(status.get("可集合竞价委托")),
        str(status.get("数据成熟度说明") or ""),
    )
    status.update(
        {
            "清单类型": "9:24预清单",
            "采集时间": datetime.now().isoformat(timespec="seconds"),
            "采集轮次": iteration,
            "建议限价上浮比例": f"{args.entry_buffer_pct * 100:.2f}%",
            "预清单说明": "基于9:20-9:24集合竞价爬升预信号，未使用9:25最终撮合/开盘价。",
        }
    )
    output_paths = export_preopen_outputs(today_ts, filtered, selected, status, queries)
    signature = selected_signature(selected)
    result = PreopenRunResult(today_ts, filtered, selected, status, queries, output_paths, signature)
    append_monitor_log(result)
    return result


def print_result(result: PreopenRunResult, previous_signature: str | None = None) -> None:
    status = result.status
    changed = previous_signature is not None and result.signature != previous_signature
    changed_text = "有变化" if changed else "无变化" if previous_signature is not None else "首次生成"
    print(
        f"[{status.get('采集时间')}] 第{status.get('采集轮次')}轮 9:24预清单: "
        f"入选{status.get('入选数', 0)} / 最终候选{status.get('最终候选数', 0)} / "
        f"成熟度={status.get('预清单数据成熟度')} / 可委托={status.get('可集合竞价委托')} / {changed_text}"
    )
    if status.get("数据成熟度说明"):
        print(f"数据说明: {status.get('数据成熟度说明')}")
    if result.selected.empty:
        print("当前无预入选标的")
        return
    keep = [
        "排序名次",
        "股票代码",
        "股票简称",
        "竞价匹配价今日",
        "建议限价上限",
        "竞价匹配金额_openapi",
        "竞价未匹配占比",
        "竞昨成交比",
        "建议权重",
    ]
    keep = [column for column in keep if column in result.selected.columns]
    print(result.selected[keep].to_string(index=False))


def same_day_at(clock: dt_time) -> datetime:
    now = datetime.now()
    return datetime.combine(now.date(), clock)


def should_run_once(args: argparse.Namespace) -> bool:
    explicit_date = explicit_trade_date(args.trade_date)
    today_now = pd.Timestamp(date.today()).normalize()
    historical_replay = explicit_date is not None and explicit_date < today_now
    return args.once or historical_replay


def main() -> None:
    args = parse_args()
    logging.info("开始运行9:24预清单监控 args=%s cwd=%s python=%s", vars(args), Path.cwd(), sys.executable)

    if should_run_once(args):
        result = run_preopen_once(args, 1)
        print_result(result)
        if args.push and not args.no_push:
            daily.send_pushplus(f"{result.trade_date.strftime('%Y-%m-%d')} 9:24预清单", build_preopen_push_content(result))
        return

    start_dt = same_day_at(args.start_time)
    end_dt = same_day_at(args.end_time)
    if start_dt > end_dt:
        raise RuntimeError("--start-time 不能晚于 --end-time")

    now = datetime.now()
    if now < start_dt:
        sleep_seconds = max(0.0, (start_dt - now).total_seconds())
        print(f"等待到 {args.start_time.strftime('%H:%M:%S')} 开始监控，约 {sleep_seconds:.0f} 秒")
        time.sleep(sleep_seconds)

    previous_signature: str | None = None
    final_result: PreopenRunResult | None = None
    iteration = 0
    while datetime.now() <= end_dt:
        iteration += 1
        result = run_preopen_once(args, iteration)
        print_result(result, previous_signature)
        previous_signature = result.signature
        final_result = result
        if args.push_each and not args.no_push:
            daily.send_pushplus(f"{result.trade_date.strftime('%Y-%m-%d')} 9:24预清单第{iteration}轮", build_preopen_push_content(result))
        sleep_until = min(args.interval_seconds, max(0.0, (end_dt - datetime.now()).total_seconds()))
        if sleep_until <= 0:
            break
        time.sleep(sleep_until)

    if final_result is None:
        final_result = run_preopen_once(args, 1)
        print_result(final_result)

    if args.push and not args.no_push:
        daily.send_pushplus(
            f"{final_result.trade_date.strftime('%Y-%m-%d')} 9:24预清单最终版",
            build_preopen_push_content(final_result),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("9:24预清单监控失败")
        print(f"9:24预清单监控失败: {exc}")
        raise SystemExit(1)
