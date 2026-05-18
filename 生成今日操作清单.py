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
DATA_ARCHIVE_DIR = BASE_DIR / "daily_data" / "竞价爬升"
ARCHIVE_SUMMARY_CSV = DATA_ARCHIVE_DIR / "竞价爬升-每日采集汇总.csv"
TRADE_RESULT_CSV = BASE_DIR / "实盘交易记录.csv"
TOP_N = 2
MIN_AUCTION_AMOUNT = 50_000_000
MIN_AUCTION_TO_YESTERDAY_RATIO = 0.022
DEFAULT_AUCTION_RATIO_MODE = "estimated"
DYNAMIC_AUCTION_RATIO_ENABLED = True
DYNAMIC_AUCTION_RATIO_STRONG_MARKET20_DIFF = 500
DYNAMIC_AUCTION_RATIO_WEAK_MARKET20_DIFF = 0
DYNAMIC_AUCTION_RATIO_STRONG = 0.020
DYNAMIC_AUCTION_RATIO_WEAK = 0.025
MIN_AUCTION_CHANGE = -9.0
MAX_AUCTION_CHANGE = None
MIN_MARKET_20_HIGH_LOW_DIFF = -600
MIN_MARKET_60_HIGH_LOW_DIFF: float | None = None
MIN_MARKET_120_HIGH_LOW_DIFF = 0
LOSS_COOLDOWN_ENABLED = True
LOSS_COOLDOWN_CONSECUTIVE_LOSSES = 2
LOSS_COOLDOWN_DAYS = 1
DEFAULT_MIN_UNMATCHED_RATIO: float | None = None
DYNAMIC_TOP_N_ENABLED = False
DYNAMIC_TOP_N_STRONG_MARKET_DIFF = 500
DYNAMIC_TOP_N_MIDDLE_MARKET_DIFF = 100
DYNAMIC_TOP_N_MIDDLE_MARKET_TOP_N = 2
DYNAMIC_TOP_N_WEAK_MARKET_TOP_N = 1
ENTRY_DAY_LOW_FROM_OPEN_RISK_EXIT = -0.05
DEFAULT_INDUSTRY_FILTER_ENABLED = False
INDUSTRY_CHANGE_MIN = 0.0
PREV_BODY_MIN = -5.0
YESTERDAY_PREV_VOLUME_RATIO_MIN = 0.4
YESTERDAY_PREV_VOLUME_RATIO_FILTER_ENABLED = True
POSITION_WEIGHT_POLICY_NAME = "强市TOP2_1_0_弱市30_70"
POSITION_WEIGHT_STRONG_MARKET20_DIFF = 250.0
POSITION_WEIGHT_MIDDLE_MARKET20_DIFF = 100.0
POSITION_WEIGHT_STRONG = (1.00, 0.00, 0.00)
POSITION_WEIGHT_MIDDLE = (0.30, 0.70, 0.0)
POSITION_WEIGHT_WEAK = (0.30, 0.70, 0.0)
POSITION_WEIGHT_AMOUNT_TILT_ENABLED = True
POSITION_WEIGHT_AMOUNT_TILT_BASE = 50_000_000
POSITION_WEIGHT_AMOUNT_TILT_POWER = 1.0
POSITION_WEIGHT_POST_NORMALIZE_CAP = 0.75
EXECUTION_ADVICE_ENABLED = True
EXECUTION_NORMAL_PREMIUM = 0.005
EXECUTION_NORMAL_PREMIUM_MAX = 0.01
EXECUTION_HIGH_PREMIUM = 0.015
EXECUTION_HIGH_PREMIUM_MAX = 0.02
EXECUTION_NO_CHASE_PREMIUM = 0.0
EXECUTION_LOW_DISCOUNT = -0.01
EXECUTION_LOW_DISCOUNT_MAX = -0.015
YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD = 20.0
VOLUME_SHAPE_RISK_FILTER_ENABLED = True
VOLUME_SHAPE_BODY0_VOL_RATIO_MIN = 5.0
VOLUME_SHAPE_BODY0_MARKET20_MIN = 200.0
VOLUME_SHAPE_BODY0_MARKET20_MAX = 1000.0
VOLUME_SHAPE_PULLBACK_YESTERDAY_BODY_MAX = -8.0
VOLUME_SHAPE_PULLBACK_PREV_BODY_MIN = 5.0
VOLUME_SHAPE_PULLBACK_MARKET20_MIN = 800.0
EARLY_WEAK_CONTINUATION_RISK_FILTER_ENABLED = True
EARLY_WEAK_CONTINUATION_M120_MAX = 80.0
EARLY_WEAK_CONTINUATION_BODY_Y_MAX = -10.0
EARLY_WEAK_CONTINUATION_CHG_MAX = -2.7
EARLY_OVERHEAT_WEAK_RISK_M20_MIN = 1000.0
EARLY_OVERHEAT_WEAK_RISK_M120_MAX = 250.0
EARLY_OVERHEAT_WEAK_RISK_BODY_Y_MAX = -6.5
EARLY_OVERHEAT_WEAK_RISK_CHG_MAX = -3.0
SECTOR_EXPLORER_SCRIPT = BASE_DIR / "竞价行业联动探索.py"
OPEN_SECTOR_EXPLORER_SCRIPT = BASE_DIR / "竞价行业开盘联动探索.py"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
PUSHPLUS_TOKEN = "e60d2f5d230f42739c52712203b9eb93"
RUN_LOG = BASE_DIR / "自动化运行日志.log"
ARCHIVE_SUMMARY_COLUMNS = [
    "交易日期",
    "昨日日期",
    "前日日期",
    "采集时间",
    "股票代码",
    "基础代码",
    "股票简称",
    "开盘价:不复权今日",
    "竞价涨幅今日",
    "竞价换手率今日",
    "竞价匹配金额_openapi",
    "竞价未匹配金额",
    "竞价未匹配占比",
    "竞价匹配价今日",
    "竞价量今日",
    "竞价未匹配量今日",
    "竞价异动类型今日",
    "集合竞价评级今日",
    "成交量昨日",
    "成交量前日",
    "实体涨跌幅昨日",
    "实体涨跌幅前日",
    "个股热度排名昨日",
    "连续涨停天数昨日",
    "昨收估算",
    "昨日成交额估算",
    "竞昨成交比估算",
    "竞昨成交比_成交金额口径",
    "竞昨成交比",
    "竞昨成交比口径",
    "昨日前日成交量比",
    "昨日前日量比风险",
    "昨日前日量比提示",
    "量价结构风险",
    "量价结构提示",
    "早期弱承接风险",
    "早期弱承接提示",
    "申万一级行业代码",
    "申万一级行业",
    "申万一级行业涨跌幅",
    "申万一级行业涨跌幅排名",
    "历史回放",
    "开仓开关",
    "市场20日高低差",
    "市场60日高低差",
    "市场120日高低差",
    "行业强度口径",
    "策略原始候选数",
    "策略配置最大入选数",
    "策略最大入选数",
    "策略动态持仓启用",
    "策略动态持仓强市场阈值",
    "策略动态持仓中市场阈值",
    "策略盘后弱承接风控阈值",
    "策略金额过滤后",
    "策略竞昨成交比口径",
    "策略竞昨成交比阈值",
    "策略竞昨成交比基础阈值",
    "策略动态竞昨阈值启用",
    "策略动态竞昨市场分层",
    "策略动态竞昨强市场20阈值",
    "策略动态竞昨弱市场20阈值",
    "策略动态竞昨强市场阈值",
    "策略动态竞昨中市场阈值",
    "策略动态竞昨弱市场阈值",
    "策略竞昨过滤后",
    "策略竞价涨幅下限",
    "策略竞价涨幅上限",
    "策略竞价涨幅过滤后",
    "策略市场20日高低差阈值",
    "策略市场60日高低差阈值",
    "策略市场120日高低差阈值",
    "策略市场宽度过滤后",
    "策略市场60过滤后",
    "策略连续亏损冷却启用",
    "策略连续亏损阈值",
    "策略连续亏损冷却天数",
    "策略连续亏损最近亏损数",
    "策略连续亏损冷却剩余交易日",
    "策略连续亏损冷却触发",
    "策略实体过滤后",
    "策略行业过滤启用",
    "策略行业过滤后",
    "策略未匹配占比阈值",
    "策略未匹配过滤后",
    "策略昨日前日成交量比下限",
    "策略昨日前日成交量比过滤启用",
    "策略昨日前日成交量比过滤后",
    "策略昨日前日成交量比过滤数",
    "策略早期弱承接风险过滤启用",
    "策略早期弱承接风险规则",
    "策略早期弱承接风险过滤后",
    "策略早期弱承接风险过滤数",
    "策略昨日前日量比风险阈值",
    "策略量价结构风险过滤启用",
    "策略量价结构风险过滤后",
    "策略量价结构风险过滤数",
    "策略量价结构风险规则",
    "策略仓位规则",
    "策略仓位市场分层",
    "策略最终候选数",
    "策略入选数",
    "策略说明",
    "是否进入策略候选池",
    "是否入选",
    "建议动作",
    "建议权重",
    "挂单建议",
    "建议挂单溢价",
    "挂单上限溢价",
    "挂单建议理由",
    "仓位规则",
    "仓位市场分层",
    "策略排序名次",
]

logging.basicConfig(
    filename=RUN_LOG,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是大于等于 1 的整数")
    return parsed


def resolve_effective_top_n(
    snapshot: dict[str, Any],
    configured_top_n: int,
    dynamic_top_n_enabled: bool = DYNAMIC_TOP_N_ENABLED,
) -> int:
    if configured_top_n < 1:
        raise ValueError("top_n 必须大于等于 1")
    if not dynamic_top_n_enabled:
        return configured_top_n

    market_diff = pd.to_numeric(pd.Series([snapshot.get("市场20日高低差")]), errors="coerce").iloc[0]
    if pd.notna(market_diff) and float(market_diff) >= DYNAMIC_TOP_N_STRONG_MARKET_DIFF:
        return configured_top_n
    if pd.notna(market_diff) and float(market_diff) >= DYNAMIC_TOP_N_MIDDLE_MARKET_DIFF:
        return min(configured_top_n, DYNAMIC_TOP_N_MIDDLE_MARKET_TOP_N)
    return min(configured_top_n, DYNAMIC_TOP_N_WEAK_MARKET_TOP_N)


def resolve_market_layer(snapshot: dict[str, Any]) -> str:
    market_diff = pd.to_numeric(pd.Series([snapshot.get("市场20日高低差")]), errors="coerce").iloc[0]
    if pd.notna(market_diff) and float(market_diff) >= DYNAMIC_TOP_N_STRONG_MARKET_DIFF:
        return "强"
    if pd.notna(market_diff) and float(market_diff) >= DYNAMIC_TOP_N_MIDDLE_MARKET_DIFF:
        return "中"
    return "弱"


def resolve_position_market_layer(snapshot: dict[str, Any]) -> str:
    market_diff = pd.to_numeric(pd.Series([snapshot.get("市场20日高低差")]), errors="coerce").iloc[0]
    if pd.notna(market_diff) and float(market_diff) >= POSITION_WEIGHT_STRONG_MARKET20_DIFF:
        return "强"
    if pd.notna(market_diff) and float(market_diff) >= POSITION_WEIGHT_MIDDLE_MARKET20_DIFF:
        return "中"
    return "弱"


def resolve_dynamic_auction_ratio_threshold(
    snapshot: dict[str, Any],
    middle_ratio: float | None,
    dynamic_enabled: bool = DYNAMIC_AUCTION_RATIO_ENABLED,
) -> tuple[str, float | None]:
    if middle_ratio is None:
        return "关闭", None
    if not dynamic_enabled:
        return "固定", middle_ratio

    market20 = pd.to_numeric(pd.Series([snapshot.get("市场20日高低差")]), errors="coerce").iloc[0]
    if pd.isna(market20):
        return "中-缺市场20", middle_ratio
    market20_value = float(market20)
    if market20_value >= DYNAMIC_AUCTION_RATIO_STRONG_MARKET20_DIFF:
        return "强", DYNAMIC_AUCTION_RATIO_STRONG
    if market20_value < DYNAMIC_AUCTION_RATIO_WEAK_MARKET20_DIFF:
        return "弱", DYNAMIC_AUCTION_RATIO_WEAK
    return "中", middle_ratio


def normalize_weight_curve(curve: tuple[float, ...], selected_count: int) -> list[float]:
    if selected_count <= 0:
        return []
    weights = list(curve[:selected_count])
    if len(weights) < selected_count:
        weights.extend([0.0] * (selected_count - len(weights)))
    clipped = [max(0.0, float(weight)) for weight in weights]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / selected_count for _ in range(selected_count)]
    return [weight / total for weight in clipped]


def resolve_position_weights(snapshot: dict[str, Any], selected_count: int) -> tuple[str, list[float]]:
    layer = resolve_position_market_layer(snapshot)
    if layer == "强":
        curve = POSITION_WEIGHT_STRONG
    elif layer == "中":
        curve = POSITION_WEIGHT_MIDDLE
    else:
        curve = POSITION_WEIGHT_WEAK
    return layer, normalize_weight_curve(curve, selected_count)


def resolve_position_weights_for_selection(
    snapshot: dict[str, Any],
    selected_df: pd.DataFrame,
) -> tuple[str, list[float]]:
    layer, base_weights = resolve_position_weights(snapshot, len(selected_df))
    if (
        selected_df.empty
        or not POSITION_WEIGHT_AMOUNT_TILT_ENABLED
        or "竞价匹配金额_openapi" not in selected_df.columns
    ):
        return layer, base_weights

    amounts = pd.to_numeric(selected_df["竞价匹配金额_openapi"], errors="coerce")
    tilted_weights: list[float] = []
    for base_weight, amount in zip(base_weights, amounts):
        if pd.isna(amount):
            multiplier = 1.0
        else:
            amount_ratio = max(float(amount) / POSITION_WEIGHT_AMOUNT_TILT_BASE, 0.1)
            multiplier = amount_ratio ** POSITION_WEIGHT_AMOUNT_TILT_POWER
        tilted_weights.append(base_weight * multiplier)
    normalized_weights = normalize_weight_curve(tuple(tilted_weights), len(tilted_weights))
    if POSITION_WEIGHT_POST_NORMALIZE_CAP is not None and len(normalized_weights) > 1:
        compressed_weights = [min(weight, POSITION_WEIGHT_POST_NORMALIZE_CAP) for weight in normalized_weights]
        normalized_weights = normalize_weight_curve(tuple(compressed_weights), len(compressed_weights))
    return layer, normalized_weights


def numeric_value(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def build_volume_ratio_risk(row: pd.Series) -> dict[str, str]:
    volume_ratio = numeric_value(row.get("昨日前日成交量比"))
    if pd.isna(volume_ratio):
        return {
            "昨日前日量比风险": "未知",
            "昨日前日量比提示": "缺少昨日前日成交量比，无法判断极端放量风险",
        }
    if volume_ratio > YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD:
        return {
            "昨日前日量比风险": "谨慎",
            "昨日前日量比提示": (
                f"昨日前日成交量比{volume_ratio:.2f}>{YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD:.1f}，"
                "昨日极端放量，回测中类似样本有接力失败风险；建议观察或降低执行优先级"
            ),
        }
    return {
        "昨日前日量比风险": "正常",
        "昨日前日量比提示": "",
    }


def describe_volume_shape_risk_rule() -> str:
    return (
        f"前日实体<=0且昨日前日量比>={VOLUME_SHAPE_BODY0_VOL_RATIO_MIN:g}且"
        f"{VOLUME_SHAPE_BODY0_MARKET20_MIN:g}<=市场20日<={VOLUME_SHAPE_BODY0_MARKET20_MAX:g}；"
        f"或昨日实体<={VOLUME_SHAPE_PULLBACK_YESTERDAY_BODY_MAX:g}且"
        f"前日实体>={VOLUME_SHAPE_PULLBACK_PREV_BODY_MIN:g}且"
        f"市场20日>={VOLUME_SHAPE_PULLBACK_MARKET20_MIN:g}"
    )


def build_volume_shape_risk(row: pd.Series) -> dict[str, str]:
    volume_ratio = numeric_value(row.get("昨日前日成交量比"))
    yesterday_body = numeric_value(row.get("实体涨跌幅昨日"))
    prev_body = numeric_value(row.get("实体涨跌幅前日"))
    market20 = numeric_value(row.get("市场20日高低差"))
    reasons: list[str] = []

    if (
        pd.notna(prev_body)
        and pd.notna(volume_ratio)
        and pd.notna(market20)
        and prev_body <= 0
        and volume_ratio >= VOLUME_SHAPE_BODY0_VOL_RATIO_MIN
        and VOLUME_SHAPE_BODY0_MARKET20_MIN <= market20 <= VOLUME_SHAPE_BODY0_MARKET20_MAX
    ):
        reasons.append(
            "前日实体<=0叠加昨日高量比，且市场20日处于中强区间，回测中易出现弱承接"
        )

    if (
        pd.notna(yesterday_body)
        and pd.notna(prev_body)
        and pd.notna(market20)
        and yesterday_body <= VOLUME_SHAPE_PULLBACK_YESTERDAY_BODY_MAX
        and prev_body >= VOLUME_SHAPE_PULLBACK_PREV_BODY_MIN
        and market20 >= VOLUME_SHAPE_PULLBACK_MARKET20_MIN
    ):
        reasons.append("强市场中前日大阳后昨日大阴，回测中次日接力赔率偏弱")

    return {
        "量价结构风险": "剔除" if reasons else "正常",
        "量价结构提示": "；".join(reasons),
    }


def describe_early_weak_continuation_risk_rule() -> str:
    return (
        f"市场120日<={EARLY_WEAK_CONTINUATION_M120_MAX:g}且昨日实体<={EARLY_WEAK_CONTINUATION_BODY_Y_MAX:g}且"
        f"竞价涨幅<={EARLY_WEAK_CONTINUATION_CHG_MAX:g}；"
        f"或市场20日>={EARLY_OVERHEAT_WEAK_RISK_M20_MIN:g}且市场120日<={EARLY_OVERHEAT_WEAK_RISK_M120_MAX:g}且"
        f"昨日实体<={EARLY_OVERHEAT_WEAK_RISK_BODY_Y_MAX:g}且竞价涨幅<={EARLY_OVERHEAT_WEAK_RISK_CHG_MAX:g}"
    )


def build_early_weak_continuation_risk(row: pd.Series) -> dict[str, str]:
    market20 = numeric_value(row.get("市场20日高低差"))
    market120 = numeric_value(row.get("市场120日高低差"))
    yesterday_body = numeric_value(row.get("实体涨跌幅昨日"))
    auction_change = numeric_value(row.get("竞价涨幅今日"))
    reasons: list[str] = []

    if (
        pd.notna(market120)
        and pd.notna(yesterday_body)
        and pd.notna(auction_change)
        and market120 <= EARLY_WEAK_CONTINUATION_M120_MAX
        and yesterday_body <= EARLY_WEAK_CONTINUATION_BODY_Y_MAX
        and auction_change <= EARLY_WEAK_CONTINUATION_CHG_MAX
    ):
        reasons.append("市场120日偏弱叠加昨日深跌且竞价偏弱，回测中容易延续弱承接")

    if (
        pd.notna(market20)
        and pd.notna(market120)
        and pd.notna(yesterday_body)
        and pd.notna(auction_change)
        and market20 >= EARLY_OVERHEAT_WEAK_RISK_M20_MIN
        and market120 <= EARLY_OVERHEAT_WEAK_RISK_M120_MAX
        and yesterday_body <= EARLY_OVERHEAT_WEAK_RISK_BODY_Y_MAX
        and auction_change <= EARLY_OVERHEAT_WEAK_RISK_CHG_MAX
    ):
        reasons.append("20日过热但120日偏弱，昨日弱承接且竞价偏弱，回测中次日延续失手概率偏高")

    return {
        "早期弱承接风险": "剔除" if reasons else "正常",
        "早期弱承接提示": "；".join(reasons),
    }


def build_execution_advice(row: pd.Series, market_layer: str) -> dict[str, Any]:
    rank = numeric_value(row.get("排序名次"))
    auction_change = numeric_value(row.get("竞价涨幅今日"))
    auction_ratio = numeric_value(row.get("竞昨成交比"))

    reasons: list[str] = []
    high_signal = False
    no_chase = False

    if market_layer == "中" and not pd.isna(auction_change) and auction_change < 0:
        return {
            "挂单建议": "挂低",
            "建议挂单溢价": EXECUTION_LOW_DISCOUNT,
            "挂单上限溢价": EXECUTION_LOW_DISCOUNT_MAX,
            "挂单建议理由": "中市场负竞价，回测挂低1%更容易改善成本且不降低成交率",
        }

    if not pd.isna(auction_change) and auction_change > 5:
        no_chase = True
        reasons.append("竞价涨幅>5%，避免高位追价")
    if not pd.isna(auction_change) and auction_change <= -3:
        no_chase = True
        reasons.append("竞价涨幅<=-3%，通常有回踩，先等价格")
    if not pd.isna(auction_ratio) and auction_ratio >= 0.030 and not pd.isna(auction_change) and auction_change < 0:
        no_chase = True
        reasons.append("竞昨成交比较高但竞价为负，先避免追高")

    if not no_chase:
        if (
            not pd.isna(auction_change)
            and 0 <= auction_change <= 2
            and not pd.isna(auction_ratio)
            and auction_ratio <= 0.030
        ):
            high_signal = True
            reasons.append("竞价涨幅0~2%且竞昨<=0.030，历史更容易开盘不给回踩")
        if not pd.isna(rank) and rank >= 2 and not pd.isna(auction_change) and 0 <= auction_change <= 5:
            high_signal = True
            reasons.append("非TOP1但竞价不弱，历史开盘即最低占比较高")
        if market_layer == "强" and not pd.isna(auction_change) and -2 <= auction_change <= 2:
            high_signal = True
            reasons.append("强市场且竞价温和，适合提高成交优先级")

    if no_chase:
        return {
            "挂单建议": "不追",
            "建议挂单溢价": EXECUTION_NO_CHASE_PREMIUM,
            "挂单上限溢价": EXECUTION_NORMAL_PREMIUM,
            "挂单建议理由": "；".join(reasons) or "信号不适合追价",
        }
    if high_signal:
        return {
            "挂单建议": "挂高",
            "建议挂单溢价": EXECUTION_HIGH_PREMIUM,
            "挂单上限溢价": EXECUTION_HIGH_PREMIUM_MAX,
            "挂单建议理由": "；".join(reasons),
        }
    return {
        "挂单建议": "普通",
        "建议挂单溢价": EXECUTION_NORMAL_PREMIUM,
        "挂单上限溢价": EXECUTION_NORMAL_PREMIUM_MAX,
        "挂单建议理由": "常规信号，优先控制滑点",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9:25 后生成今日操作清单")
    parser.add_argument(
        "--trade-date",
        help="指定交易日期，格式 YYYYMMDD，用于历史回放或盘后验证",
    )
    parser.add_argument("--push", action="store_true", help="历史回放时也发送 PushPlus 推送")
    parser.add_argument("--no-push", action="store_true", help="本次运行不发送 PushPlus 推送")
    parser.add_argument(
        "--top-n",
        type=positive_int,
        default=TOP_N,
        help="配置最大入选标的数量，默认 2；动态持仓开启时按市场强弱自动降到 2 或 1",
    )
    top_mode_group = parser.add_mutually_exclusive_group()
    top_mode_group.add_argument(
        "--fixed-top-n",
        dest="fixed_top_n",
        action="store_true",
        default=not DYNAMIC_TOP_N_ENABLED,
        help="关闭动态持仓，严格使用 --top-n 指定的最大入选数",
    )
    top_mode_group.add_argument(
        "--dynamic-top-n",
        dest="fixed_top_n",
        action="store_false",
        help="启用动态持仓，按市场强弱自动降低入选数",
    )
    parser.add_argument(
        "--min-auction-ratio",
        type=float,
        default=MIN_AUCTION_TO_YESTERDAY_RATIO,
        help="竞昨成交比中市场阈值，默认 0.022；动态竞昨开启时强市场用0.020、弱市场用0.025",
    )
    auction_ratio_mode_group = parser.add_mutually_exclusive_group()
    auction_ratio_mode_group.add_argument(
        "--dynamic-auction-ratio",
        dest="dynamic_auction_ratio",
        action="store_true",
        default=DYNAMIC_AUCTION_RATIO_ENABLED,
        help="启用市场20日分层动态竞昨阈值，默认开启",
    )
    auction_ratio_mode_group.add_argument(
        "--fixed-auction-ratio",
        dest="dynamic_auction_ratio",
        action="store_false",
        help="关闭市场20日分层，固定使用 --min-auction-ratio",
    )
    parser.add_argument(
        "--auction-ratio-mode",
        choices=["estimated", "amount"],
        default=DEFAULT_AUCTION_RATIO_MODE,
        help="竞昨成交比口径：estimated=用开盘价/竞价涨幅/昨日成交量估算昨日成交额；amount=直接用昨日成交金额。默认 estimated",
    )
    parser.add_argument(
        "--no-auction-ratio-filter",
        action="store_true",
        help="关闭竞昨成交比过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--min-auction-change",
        type=float,
        default=MIN_AUCTION_CHANGE,
        help="竞价涨幅最低阈值，默认 -9；与 --max-auction-change 组成增强版过滤",
    )
    parser.add_argument(
        "--max-auction-change",
        type=float,
        default=MAX_AUCTION_CHANGE,
        help="竞价涨幅最高阈值，默认不限；如需回放旧口径可传 6",
    )
    parser.add_argument(
        "--no-auction-change-filter",
        action="store_true",
        help="关闭竞价涨幅过滤，仅用于回放对照旧策略",
    )
    parser.add_argument(
        "--min-market20-high-low-diff",
        type=float,
        default=MIN_MARKET_20_HIGH_LOW_DIFF,
        help="市场20日新高数-新低数最低阈值，默认 -600；当前主口径过滤短期极弱环境",
    )
    parser.add_argument(
        "--no-market20-filter",
        action="store_true",
        help="关闭市场20日高低差过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--min-market60-high-low-diff",
        type=float,
        default=MIN_MARKET_60_HIGH_LOW_DIFF,
        help="市场60日新高数-新低数最低阈值，默认关闭；如需回放旧口径可传 50",
    )
    parser.add_argument(
        "--min-market120-high-low-diff",
        type=float,
        default=MIN_MARKET_120_HIGH_LOW_DIFF,
        help="市场120日新高数-新低数最低阈值，默认 0；当前主口径确认中期市场环境",
    )
    parser.add_argument(
        "--prev-body-min",
        type=float,
        default=PREV_BODY_MIN,
        help="前日实体涨跌幅最低阈值，默认 -5",
    )
    parser.add_argument(
        "--no-market60-filter",
        action="store_true",
        help="关闭市场60日高低差过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--no-market120-filter",
        action="store_true",
        help="关闭市场120日高低差过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--trade-result-file",
        default=str(TRADE_RESULT_CSV),
        help="已完成实盘交易记录CSV，用于连续亏损冷却建议；默认 实盘交易记录.csv",
    )
    parser.add_argument(
        "--loss-cooldown-consecutive-losses",
        type=positive_int,
        default=LOSS_COOLDOWN_CONSECUTIVE_LOSSES,
        help="连续亏损多少笔后给出冷却建议，默认 2",
    )
    parser.add_argument(
        "--loss-cooldown-days",
        type=positive_int,
        default=LOSS_COOLDOWN_DAYS,
        help="连续亏损触发后建议冷却多少个交易日，默认 1",
    )
    parser.add_argument(
        "--no-loss-cooldown",
        action="store_true",
        default=not LOSS_COOLDOWN_ENABLED,
        help="关闭连续亏损冷却建议，仅用于回放对照",
    )
    parser.add_argument(
        "--min-unmatched-ratio",
        type=float,
        default=DEFAULT_MIN_UNMATCHED_RATIO,
        help="竞价未匹配占比最低阈值，默认不启用；可传 0.005 回放近期增强版本",
    )
    parser.add_argument(
        "--min-prevday-volume-ratio",
        type=float,
        default=YESTERDAY_PREV_VOLUME_RATIO_MIN,
        help="昨日前日成交量比最低阈值，默认 0.4；低于该值的样本直接剔除",
    )
    parser.add_argument(
        "--no-prevday-volume-ratio-filter",
        action="store_true",
        default=not YESTERDAY_PREV_VOLUME_RATIO_FILTER_ENABLED,
        help="关闭昨日前日成交量比硬过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--no-volume-shape-risk-filter",
        action="store_true",
        default=not VOLUME_SHAPE_RISK_FILTER_ENABLED,
        help="关闭量价结构风险硬过滤，仅用于回放对照",
    )
    parser.add_argument(
        "--no-early-weak-continuation-filter",
        action="store_true",
        default=not EARLY_WEAK_CONTINUATION_RISK_FILTER_ENABLED,
        help="关闭早期弱承接硬过滤，仅用于回放对照",
    )
    industry_group = parser.add_mutually_exclusive_group()
    industry_group.add_argument(
        "--industry-filter",
        dest="industry_filter",
        action="store_true",
        default=DEFAULT_INDUSTRY_FILTER_ENABLED,
        help="启用申万一级行业涨幅>0过滤，偏保守回放/运行时使用",
    )
    industry_group.add_argument(
        "--no-industry-filter",
        dest="industry_filter",
        action="store_false",
        help="关闭申万一级行业涨幅过滤，当前默认",
    )
    parser.add_argument(
        "--no-execution-advice",
        action="store_true",
        help="不输出挂单建议列，仅保留选股和仓位结果",
    )
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


def missing_market_snapshot_fields(
    snapshot: dict[str, Any],
    require_market60: bool = MIN_MARKET_60_HIGH_LOW_DIFF is not None,
    require_market120: bool = MIN_MARKET_120_HIGH_LOW_DIFF is not None,
) -> list[str]:
    missing: list[str] = []
    if "日期" not in snapshot or pd.isna(pd.to_datetime(snapshot.get("日期"), errors="coerce")):
        missing.append("日期")
    required_fields = ["市场20日高低差"]
    if require_market60:
        required_fields.append("市场60日高低差")
    if require_market120:
        required_fields.append("市场120日高低差")
    missing.extend(
        field
        for field in required_fields
        if field not in snapshot
        or pd.isna(pd.to_numeric(pd.Series([snapshot.get(field)]), errors="coerce").iloc[0])
    )
    return missing


def ensure_fresh_snapshot(
    snapshot: dict[str, Any],
    required_date: pd.Timestamp | None,
    require_market60: bool = MIN_MARKET_60_HIGH_LOW_DIFF is not None,
    require_market120: bool = MIN_MARKET_120_HIGH_LOW_DIFF is not None,
) -> dict[str, Any]:
    missing_fields = missing_market_snapshot_fields(snapshot, require_market60, require_market120)
    if required_date is None and not missing_fields:
        return snapshot
    snapshot_date_value = pd.to_datetime(snapshot.get("日期"), errors="coerce")
    snapshot_date = pd.Timestamp("1900-01-01") if pd.isna(snapshot_date_value) else pd.Timestamp(snapshot_date_value).normalize()
    required_date = required_date.normalize() if required_date is not None else None
    if required_date is not None and snapshot_date >= required_date and not missing_fields:
        return snapshot

    logging.info(
        "市场快照过期或缺字段，尝试刷新 snapshot_date=%s required_date=%s missing_fields=%s",
        snapshot_date.strftime("%Y-%m-%d"),
        required_date.strftime("%Y-%m-%d") if required_date is not None else None,
        missing_fields,
    )
    refresh_market_snapshot()
    snapshot = read_snapshot(MARKET_SNAPSHOT_JSON)
    snapshot_date = pd.Timestamp(snapshot["日期"]).normalize()
    missing_fields = missing_market_snapshot_fields(snapshot, require_market60, require_market120)
    if (required_date is not None and snapshot_date < required_date) or missing_fields:
        raise RuntimeError(
            f"市场宽度快照不可用: 当前 {snapshot_date.strftime('%Y-%m-%d')}，"
            f"需要至少 {required_date.strftime('%Y-%m-%d') if required_date is not None else '不限'}，"
            f"缺字段 {missing_fields}。请检查 获取市场宽度.py 或数据源。"
        )
    return snapshot


def read_market_snapshot(
    trade_date: pd.Timestamp,
    historical_replay: bool,
    required_date: pd.Timestamp | None = None,
    require_market60: bool = MIN_MARKET_60_HIGH_LOW_DIFF is not None,
    require_market120: bool = MIN_MARKET_120_HIGH_LOW_DIFF is not None,
) -> dict[str, Any]:
    if not historical_replay:
        return ensure_fresh_snapshot(read_snapshot(MARKET_SNAPSHOT_JSON), required_date, require_market60, require_market120)

    if not MARKET_HISTORY_CSV.exists():
        raise FileNotFoundError(f"未找到市场宽度历史文件: {MARKET_HISTORY_CSV}")

    history_df = pd.read_csv(MARKET_HISTORY_CSV, encoding="utf-8-sig")
    required_columns = ["日期", "市场20日高低差"]
    if require_market60:
        required_columns.append("市场60日高低差")
    if require_market120:
        required_columns.append("市场120日高低差")
    if any(column not in history_df.columns for column in required_columns):
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
        "开仓开关": "通过",
        "规则": "市场20日高低差>=-600 且 市场120日高低差>=0；市场20日高低差同时用于动态竞昨阈值",
        "数据来源": str(MARKET_HISTORY_CSV),
    }
    if "市场60日高低差" in row.index and pd.notna(row["市场60日高低差"]):
        diff60 = int(pd.to_numeric(row["市场60日高低差"], errors="raise"))
        snapshot["市场60日高低差"] = diff60
    if "市场120日高低差" in row.index and pd.notna(row["市场120日高低差"]):
        snapshot["市场120日高低差"] = int(pd.to_numeric(row["市场120日高低差"], errors="raise"))
    market_switch_passed = True
    if MIN_MARKET_20_HIGH_LOW_DIFF is not None and diff20 < MIN_MARKET_20_HIGH_LOW_DIFF:
        market_switch_passed = False
    if (
        MIN_MARKET_60_HIGH_LOW_DIFF is not None
        and snapshot.get("市场60日高低差") is not None
        and snapshot["市场60日高低差"] < MIN_MARKET_60_HIGH_LOW_DIFF
    ):
        market_switch_passed = False
    if (
        MIN_MARKET_120_HIGH_LOW_DIFF is not None
        and snapshot.get("市场120日高低差") is not None
        and snapshot["市场120日高低差"] < MIN_MARKET_120_HIGH_LOW_DIFF
    ):
        market_switch_passed = False
    snapshot["开仓开关"] = "通过" if market_switch_passed else "不通过"
    for column in [
        "市场20日新高数",
        "市场20日新低数",
        "市场60日新高数",
        "市场60日新低数",
        "市场120日新高数",
        "市场120日新低数",
    ]:
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


def resolve_local_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else BASE_DIR / path


def parse_trade_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(series, errors="coerce")
    return parsed.fillna(fallback).dt.normalize()


def parse_return_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    is_percent = text.str.endswith("%", na=False)
    cleaned = text.str.replace("%", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    values = values.where(~is_percent, values / 100)
    values = values.where(values.abs() <= 1, values / 100)
    return values


def build_loss_cooldown_status(
    trade_date: pd.Timestamp,
    trade_result_file: str | Path = TRADE_RESULT_CSV,
    enabled: bool = LOSS_COOLDOWN_ENABLED,
    consecutive_losses: int = LOSS_COOLDOWN_CONSECUTIVE_LOSSES,
    cooldown_days: int = LOSS_COOLDOWN_DAYS,
    calendar_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    trade_result_path = resolve_local_path(trade_result_file)
    status: dict[str, Any] = {
        "启用": enabled,
        "阈值": consecutive_losses,
        "冷却天数": cooldown_days,
        "交易记录文件": str(trade_result_path),
        "文件存在": trade_result_path.exists(),
        "有效记录数": 0,
        "最近亏损数": 0,
        "触发日期": None,
        "冷却剩余交易日": 0,
        "触发": False,
        "说明": "连续亏损冷却建议关闭" if not enabled else "未触发连续亏损冷却建议",
    }
    if not enabled:
        return status
    if consecutive_losses < 1 or cooldown_days < 1:
        raise ValueError("连续亏损冷却建议阈值和建议天数必须大于等于 1")
    if not trade_result_path.exists():
        status["说明"] = f"交易记录文件不存在，跳过连续亏损冷却建议: {trade_result_path.name}"
        return status

    records = pd.read_csv(trade_result_path, encoding="utf-8-sig")
    if "观察状态" in records.columns:
        records = records[records["观察状态"].astype(str).isin(["已卖出", "已完成", "closed", "Closed"])].copy()
    return_column = next((column for column in ["单笔净收益率", "收益率", "净收益率", "盈亏比例"] if column in records.columns), None)
    if return_column is None:
        status["说明"] = f"交易记录缺少收益率字段，跳过连续亏损冷却建议: {trade_result_path.name}"
        return status

    date_column = next((column for column in ["卖出日期", "平仓日期", "交易日期", "日期"] if column in records.columns), None)
    records = records.copy()
    records["_收益率"] = parse_return_series(records[return_column])
    records["_原始顺序"] = range(len(records))
    if date_column is not None:
        records["_卖出日期"] = parse_trade_date_series(records[date_column])
        records = records[records["_卖出日期"].notna() & (records["_卖出日期"] < trade_date.normalize())].copy()
    else:
        records["_卖出日期"] = pd.NaT
    records = records[records["_收益率"].notna()].copy()
    if records.empty:
        status["说明"] = "交易记录无可用已完成收益，未触发连续亏损冷却建议"
        return status
    sort_columns = ["_卖出日期", "_原始顺序"] if date_column is not None else ["_原始顺序"]
    records = records.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    status["有效记录数"] = int(len(records))

    loss_streak = 0
    last_trigger_date: pd.Timestamp | None = None
    last_event_was_trigger = False
    for _, record in records.iterrows():
        if float(record["_收益率"]) < 0:
            loss_streak += 1
            last_event_was_trigger = False
        else:
            loss_streak = 0
            last_event_was_trigger = False
        if loss_streak >= consecutive_losses:
            last_trigger_date = pd.Timestamp(record["_卖出日期"]).normalize() if pd.notna(record["_卖出日期"]) else None
            loss_streak = 0
            last_event_was_trigger = True
    status["最近亏损数"] = int(consecutive_losses if last_event_was_trigger else loss_streak)
    if last_trigger_date is None:
        status["说明"] = f"最近连续亏损 {loss_streak} 笔，未达到 {consecutive_losses} 笔冷却阈值"
        return status

    status["触发日期"] = last_trigger_date.strftime("%Y-%m-%d")
    if date_column is None:
        status["说明"] = "交易记录缺少卖出日期，无法判断冷却交易日"
        return status

    calendar = calendar_df.copy() if calendar_df is not None else load_trade_calendar()
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"], errors="coerce").dt.normalize()
    trade_days = calendar.loc[
        (calendar["trade_date"] > last_trigger_date) & (calendar["trade_date"] <= trade_date.normalize()),
        "trade_date",
    ]
    elapsed_days = int(len(trade_days))
    if 1 <= elapsed_days <= cooldown_days:
        remaining_days = cooldown_days - elapsed_days + 1
        status["触发"] = True
        status["冷却剩余交易日"] = remaining_days
        status["说明"] = (
            f"{last_trigger_date.strftime('%Y-%m-%d')} 触发连续{consecutive_losses}亏，"
            f"今日处于第 {elapsed_days}/{cooldown_days} 个冷却交易日"
        )
    else:
        status["说明"] = (
            f"{last_trigger_date.strftime('%Y-%m-%d')} 曾触发连续{consecutive_losses}亏，"
            f"已过 {elapsed_days} 个交易日，冷却结束"
        )
    return status


def cn_date(ts: pd.Timestamp) -> str:
    # Historical replay must include the year; otherwise iwencai may resolve
    # "1月16日" to the current year and silently return the wrong date.
    return f"{ts.year}年{ts.month}月{ts.day}日"


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
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
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
    return merged_df, sector_df, sector_source


def to_json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=to_json_safe),
        encoding="utf-8",
    )


def build_archive_record_frame(
    trade_date: pd.Timestamp,
    prev_date: pd.Timestamp,
    prev2_date: pd.Timestamp,
    merged_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    snapshot: dict[str, Any],
    status: dict[str, Any],
    sector_source: str,
    historical_replay: bool,
) -> pd.DataFrame:
    out = merged_df.copy()
    out.insert(0, "交易日期", date_token(trade_date))
    out.insert(1, "昨日日期", date_token(prev_date))
    out.insert(2, "前日日期", date_token(prev2_date))
    out.insert(3, "采集时间", datetime.now().isoformat(timespec="seconds"))
    out["历史回放"] = historical_replay
    out["开仓开关"] = status.get("开仓开关", snapshot.get("开仓开关"))
    out["市场20日高低差"] = snapshot.get("市场20日高低差")
    out["市场60日高低差"] = snapshot.get("市场60日高低差")
    out["市场120日高低差"] = snapshot.get("市场120日高低差")
    out["行业强度口径"] = sector_source
    out["策略原始候选数"] = status.get("原始候选数")
    out["策略配置最大入选数"] = status.get("配置最大入选数")
    out["策略最大入选数"] = status.get("最大入选数")
    out["策略动态持仓启用"] = status.get("动态持仓启用")
    out["策略动态持仓强市场阈值"] = status.get("动态持仓强市场阈值")
    out["策略动态持仓中市场阈值"] = status.get("动态持仓中市场阈值")
    out["策略盘后弱承接风控阈值"] = status.get("盘后弱承接风控阈值")
    out["策略金额过滤后"] = status.get("金额过滤后")
    out["策略竞昨成交比口径"] = status.get("竞昨成交比口径")
    out["策略竞昨成交比阈值"] = status.get("竞昨成交比阈值")
    out["策略竞昨成交比基础阈值"] = status.get("竞昨成交比基础阈值")
    out["策略动态竞昨阈值启用"] = status.get("动态竞昨阈值启用")
    out["策略动态竞昨市场分层"] = status.get("动态竞昨市场分层")
    out["策略动态竞昨强市场20阈值"] = status.get("动态竞昨强市场20阈值")
    out["策略动态竞昨弱市场20阈值"] = status.get("动态竞昨弱市场20阈值")
    out["策略动态竞昨强市场阈值"] = status.get("动态竞昨强市场阈值")
    out["策略动态竞昨中市场阈值"] = status.get("动态竞昨中市场阈值")
    out["策略动态竞昨弱市场阈值"] = status.get("动态竞昨弱市场阈值")
    out["策略竞昨过滤后"] = status.get("竞昨过滤后")
    out["策略竞价涨幅下限"] = status.get("竞价涨幅下限")
    out["策略竞价涨幅上限"] = status.get("竞价涨幅上限")
    out["策略竞价涨幅过滤后"] = status.get("竞价涨幅过滤后")
    out["策略市场20日高低差阈值"] = status.get("市场20日高低差阈值")
    out["策略市场60日高低差阈值"] = status.get("市场60日高低差阈值")
    out["策略市场120日高低差阈值"] = status.get("市场120日高低差阈值")
    out["策略市场宽度过滤后"] = status.get("市场宽度过滤后")
    out["策略市场60过滤后"] = status.get("市场60过滤后")
    out["策略连续亏损冷却启用"] = status.get("连续亏损冷却启用")
    out["策略连续亏损阈值"] = status.get("连续亏损阈值")
    out["策略连续亏损冷却天数"] = status.get("连续亏损冷却天数")
    out["策略连续亏损最近亏损数"] = status.get("连续亏损最近亏损数")
    out["策略连续亏损冷却剩余交易日"] = status.get("连续亏损冷却剩余交易日")
    out["策略连续亏损冷却触发"] = status.get("连续亏损冷却触发")
    out["策略实体过滤后"] = status.get("实体过滤后")
    out["策略行业过滤启用"] = status.get("行业过滤启用")
    out["策略行业过滤后"] = status.get("行业过滤后")
    out["策略未匹配占比阈值"] = status.get("未匹配占比阈值")
    out["策略未匹配过滤后"] = status.get("未匹配过滤后")
    out["策略昨日前日成交量比下限"] = status.get("昨日前日成交量比下限")
    out["策略昨日前日成交量比过滤启用"] = status.get("昨日前日成交量比过滤启用")
    out["策略昨日前日成交量比过滤后"] = status.get("昨日前日成交量比过滤后")
    out["策略昨日前日成交量比过滤数"] = status.get("昨日前日成交量比过滤数")
    out["策略早期弱承接风险过滤启用"] = status.get("早期弱承接风险过滤启用")
    out["策略早期弱承接风险规则"] = status.get("早期弱承接风险规则")
    out["策略早期弱承接风险过滤后"] = status.get("早期弱承接风险过滤后")
    out["策略早期弱承接风险过滤数"] = status.get("早期弱承接风险过滤数")
    out["策略昨日前日量比风险阈值"] = status.get("昨日前日量比风险阈值")
    out["策略量价结构风险过滤启用"] = status.get("量价结构风险过滤启用")
    out["策略量价结构风险过滤后"] = status.get("量价结构风险过滤后")
    out["策略量价结构风险过滤数"] = status.get("量价结构风险过滤数")
    out["策略量价结构风险规则"] = status.get("量价结构风险规则")
    out["策略仓位规则"] = status.get("仓位规则")
    out["策略仓位市场分层"] = status.get("仓位市场分层")
    out["策略最终候选数"] = status.get("最终候选数")
    out["策略入选数"] = status.get("入选数")
    out["策略说明"] = status.get("结果说明")

    filtered_rank = (
        filtered_df.set_index("基础代码")["排序名次"].to_dict()
        if not filtered_df.empty and {"基础代码", "排序名次"}.issubset(filtered_df.columns)
        else {}
    )
    selected_codes = set(selected_df["基础代码"]) if "基础代码" in selected_df.columns else set()
    selected_actions = (
        selected_df.set_index("基础代码")["建议动作"].to_dict()
        if not selected_df.empty and {"基础代码", "建议动作"}.issubset(selected_df.columns)
        else {}
    )
    selected_weights = (
        selected_df.set_index("基础代码")["建议权重"].to_dict()
        if not selected_df.empty and {"基础代码", "建议权重"}.issubset(selected_df.columns)
        else {}
    )
    selected_execution_advice = (
        selected_df.set_index("基础代码")["挂单建议"].to_dict()
        if not selected_df.empty and {"基础代码", "挂单建议"}.issubset(selected_df.columns)
        else {}
    )
    selected_execution_premium = (
        selected_df.set_index("基础代码")["建议挂单溢价"].to_dict()
        if not selected_df.empty and {"基础代码", "建议挂单溢价"}.issubset(selected_df.columns)
        else {}
    )
    selected_execution_premium_max = (
        selected_df.set_index("基础代码")["挂单上限溢价"].to_dict()
        if not selected_df.empty and {"基础代码", "挂单上限溢价"}.issubset(selected_df.columns)
        else {}
    )
    selected_execution_reason = (
        selected_df.set_index("基础代码")["挂单建议理由"].to_dict()
        if not selected_df.empty and {"基础代码", "挂单建议理由"}.issubset(selected_df.columns)
        else {}
    )
    candidate_volume_risks = (
        filtered_df.set_index("基础代码")["昨日前日量比风险"].to_dict()
        if not filtered_df.empty and {"基础代码", "昨日前日量比风险"}.issubset(filtered_df.columns)
        else {}
    )
    candidate_volume_risk_notes = (
        filtered_df.set_index("基础代码")["昨日前日量比提示"].to_dict()
        if not filtered_df.empty and {"基础代码", "昨日前日量比提示"}.issubset(filtered_df.columns)
        else {}
    )
    candidate_shape_risks = (
        filtered_df.set_index("基础代码")["量价结构风险"].to_dict()
        if not filtered_df.empty and {"基础代码", "量价结构风险"}.issubset(filtered_df.columns)
        else {}
    )
    candidate_shape_risk_notes = (
        filtered_df.set_index("基础代码")["量价结构提示"].to_dict()
        if not filtered_df.empty and {"基础代码", "量价结构提示"}.issubset(filtered_df.columns)
        else {}
    )
    candidate_early_weak_risks = (
        filtered_df.set_index("基础代码")["早期弱承接风险"].to_dict()
        if not filtered_df.empty and {"基础代码", "早期弱承接风险"}.issubset(filtered_df.columns)
        else {}
    )
    candidate_early_weak_risk_notes = (
        filtered_df.set_index("基础代码")["早期弱承接提示"].to_dict()
        if not filtered_df.empty and {"基础代码", "早期弱承接提示"}.issubset(filtered_df.columns)
        else {}
    )
    selected_weight_rules = (
        selected_df.set_index("基础代码")["仓位规则"].to_dict()
        if not selected_df.empty and {"基础代码", "仓位规则"}.issubset(selected_df.columns)
        else {}
    )
    selected_weight_layers = (
        selected_df.set_index("基础代码")["仓位市场分层"].to_dict()
        if not selected_df.empty and {"基础代码", "仓位市场分层"}.issubset(selected_df.columns)
        else {}
    )
    out["是否进入策略候选池"] = out["基础代码"].isin(filtered_rank.keys())
    out["是否入选"] = out["基础代码"].isin(selected_codes)
    out["建议动作"] = out["基础代码"].map(selected_actions)
    out["建议权重"] = out["基础代码"].map(selected_weights)
    out["挂单建议"] = out["基础代码"].map(selected_execution_advice)
    out["建议挂单溢价"] = out["基础代码"].map(selected_execution_premium)
    out["挂单上限溢价"] = out["基础代码"].map(selected_execution_premium_max)
    out["挂单建议理由"] = out["基础代码"].map(selected_execution_reason)
    out["昨日前日量比风险"] = out["基础代码"].map(candidate_volume_risks)
    out["昨日前日量比提示"] = out["基础代码"].map(candidate_volume_risk_notes)
    out["量价结构风险"] = out["基础代码"].map(candidate_shape_risks)
    out["量价结构提示"] = out["基础代码"].map(candidate_shape_risk_notes)
    out["早期弱承接风险"] = out["基础代码"].map(candidate_early_weak_risks)
    out["早期弱承接提示"] = out["基础代码"].map(candidate_early_weak_risk_notes)
    out["仓位规则"] = out["基础代码"].map(selected_weight_rules)
    out["仓位市场分层"] = out["基础代码"].map(selected_weight_layers)
    out["策略排序名次"] = out["基础代码"].map(filtered_rank)
    return out


def upsert_archive_summary(record_df: pd.DataFrame, trade_date: pd.Timestamp) -> None:
    ARCHIVE_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    date_text = date_token(trade_date)
    summary_record_df = record_df[[column for column in ARCHIVE_SUMMARY_COLUMNS if column in record_df.columns]].copy()
    if ARCHIVE_SUMMARY_CSV.exists():
        old_df = pd.read_csv(ARCHIVE_SUMMARY_CSV, encoding="utf-8-sig")
        if "交易日期" in old_df.columns:
            old_df["交易日期"] = old_df["交易日期"].astype(str)
            old_df = old_df[old_df["交易日期"] != date_text].copy()
        if old_df.empty:
            summary_df = summary_record_df
        else:
            summary_df = pd.concat([old_df, summary_record_df], ignore_index=True, sort=False)
    else:
        summary_df = summary_record_df
    summary_df["交易日期"] = summary_df["交易日期"].astype(str)
    if "基础代码" in summary_df.columns:
        summary_df["基础代码"] = summary_df["基础代码"].astype(str).str.zfill(6)
    summary_df = summary_df.sort_values(
        ["交易日期", "是否入选", "策略排序名次", "基础代码"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    summary_df.to_csv(ARCHIVE_SUMMARY_CSV, index=False, encoding="utf-8-sig")


def archive_daily_data(
    trade_date: pd.Timestamp,
    prev_date: pd.Timestamp,
    prev2_date: pd.Timestamp,
    raw_frames: dict[str, pd.DataFrame],
    standardized_frames: dict[str, pd.DataFrame],
    sector_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    snapshot: dict[str, Any],
    status: dict[str, Any],
    queries: dict[str, str],
    date_map: dict[str, str],
    sector_source: str,
    historical_replay: bool,
) -> Path:
    date_text = date_token(trade_date)
    archive_dir = DATA_ARCHIVE_DIR / date_text
    archive_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, Any]] = {}

    def save_csv(filename: str, df: pd.DataFrame) -> None:
        path = archive_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files[filename] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

    for label, frame in raw_frames.items():
        save_csv(f"问财原始_{label}.csv", frame)
    for label, frame in standardized_frames.items():
        save_csv(f"问财标准化_{label}.csv", frame)

    save_csv("申万一级行业快照.csv", sector_df)
    record_df = build_archive_record_frame(
        trade_date,
        prev_date,
        prev2_date,
        merged_df,
        filtered_df,
        selected_df,
        snapshot,
        status,
        sector_source,
        historical_replay,
    )
    save_csv("全量候选_含行业因子.csv", merged_df)
    save_csv("每日采集记录.csv", record_df)
    save_csv("策略候选池.csv", filtered_df)
    save_csv("入选清单.csv", selected_df)
    upsert_archive_summary(record_df, trade_date)

    write_json(archive_dir / "市场快照.json", snapshot)
    write_json(
        archive_dir / "运行摘要.json",
        {
            "schema_version": 1,
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "prev_date": prev_date.strftime("%Y-%m-%d"),
            "prev2_date": prev2_date.strftime("%Y-%m-%d"),
            "historical_replay": historical_replay,
            "configured_top_n": status.get("配置最大入选数", TOP_N),
            "top_n": status.get("最大入选数", TOP_N),
            "dynamic_top_n_enabled": status.get("动态持仓启用", DYNAMIC_TOP_N_ENABLED),
            "dynamic_top_n_strong_market_diff": DYNAMIC_TOP_N_STRONG_MARKET_DIFF,
            "dynamic_top_n_middle_market_diff": DYNAMIC_TOP_N_MIDDLE_MARKET_DIFF,
            "entry_day_low_from_open_risk_exit": status.get("盘后弱承接风控阈值", ENTRY_DAY_LOW_FROM_OPEN_RISK_EXIT),
            "industry_filter_enabled": status.get("行业过滤启用", DEFAULT_INDUSTRY_FILTER_ENABLED),
            "industry_change_min": INDUSTRY_CHANGE_MIN,
            "min_auction_change": status.get("竞价涨幅下限"),
            "max_auction_change": status.get("竞价涨幅上限"),
            "min_market20_high_low_diff": status.get("市场20日高低差阈值"),
            "min_market60_high_low_diff": status.get("市场60日高低差阈值"),
            "min_market120_high_low_diff": status.get("市场120日高低差阈值"),
            "loss_cooldown_enabled": status.get("连续亏损冷却启用"),
            "loss_cooldown_consecutive_losses": status.get("连续亏损阈值"),
            "loss_cooldown_days": status.get("连续亏损冷却天数"),
            "loss_cooldown_triggered": status.get("连续亏损冷却触发"),
            "position_weight_policy": status.get("仓位规则", POSITION_WEIGHT_POLICY_NAME),
            "position_weight_layer": status.get("仓位市场分层"),
            "min_unmatched_ratio": status.get("未匹配占比阈值"),
            "min_prevday_volume_ratio": status.get("昨日前日成交量比下限"),
            "prevday_volume_ratio_filter_enabled": status.get("昨日前日成交量比过滤启用"),
            "prev_body_min": status.get("前日实体阈值", PREV_BODY_MIN),
            "sector_source": sector_source,
            "status": status,
            "queries": queries,
            "date_map": date_map,
            "files": files,
            "summary_csv": str(ARCHIVE_SUMMARY_CSV),
        },
    )
    logging.info("每日数据归档完成 trade_date=%s archive_dir=%s", date_text, archive_dir)
    return archive_dir


def build_queries(today_ts: pd.Timestamp, prev_ts: pd.Timestamp, prev2_ts: pd.Timestamp) -> dict[str, str]:
    today_cn = cn_date(today_ts)
    prev_cn = cn_date(prev_ts)
    prev2_cn = cn_date(prev2_ts)

    base_query = (
        f"{today_cn}9点25分最低价>{today_cn}9点24分最高价，"
        f"{today_cn}9点24分最低价>={today_cn}9点23分最高价，"
        f"{today_cn}竞价涨幅，{today_cn}竞价换手率，"
        f"{today_cn}上市天数大于3，{prev_cn}个股热度排名前100"
    )
    base_tail_query = (
        f"{today_cn}9点23分最低价>={today_cn}9点22分最高价，"
        f"{today_cn}9点22分最低价>={today_cn}9点21分最高价，"
        f"{today_cn}9点21分最低价>={today_cn}9点20分最高价，"
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
    return {"base": base_query, "base_tail": base_tail_query, "detail": detail_query, "amount": amount_query}


def replace_date_markers(columns: pd.Index, mapping: dict[str, str]) -> pd.Index:
    updated = columns.astype(str)
    for token, label in mapping.items():
        updated = updated.str.replace(rf"\[{token}\]", label, regex=True)
        updated = updated.str.replace(rf"\[{token}(?=\s)", f"[{label}", regex=True)
    return updated


def pick_column(
    columns: list[str],
    keywords: list[str],
    exclude_patterns: list[str] | None = None,
) -> str | None:
    exclude_patterns = exclude_patterns or []

    def is_allowed(column: str) -> bool:
        return not any(re.search(pattern, column) for pattern in exclude_patterns)

    for keyword in keywords:
        exact = [column for column in columns if column == keyword and is_allowed(column)]
        if exact:
            return exact[0]
    for keyword in keywords:
        partial = [column for column in columns if keyword in column and is_allowed(column)]
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
        "竞价换手率今日": ["竞价换手率今日", "竞价换手率", "分时换手率今日", "分时换手率["],
        "竞价匹配金额_openapi": ["竞价匹配金额_openapi", "竞价金额今日", "竞价金额"],
        "竞价未匹配金额": ["竞价未匹配金额"],
        "竞价匹配价今日": ["竞价匹配价今日", "竞价匹配价", "集合竞价匹配价"],
        "竞价量今日": ["竞价量今日", "竞价量"],
        "竞价未匹配量今日": ["竞价未匹配量今日", "竞价未匹配量"],
        "开盘价:不复权今日": ["开盘价:不复权今日", "开盘价今日", "开盘价:不复权"],
        "成交金额昨日": ["成交金额昨日", "成交额昨日", "成交金额", "成交额"],
        "成交量昨日": ["成交量昨日", "成交量"],
        "成交量前日": ["成交量前日"],
        "实体涨跌幅昨日": ["实体涨跌幅昨日", "实体涨跌幅"],
        "实体涨跌幅前日": ["实体涨跌幅前日"],
        "个股热度排名昨日": ["个股热度排名昨日", "个股热度排名", "个股热度排名前100"],
        "连续涨停天数昨日": ["连续涨停天数昨日", "连续涨停天数"],
    }
    date_sensitive_targets = {
        "竞价涨幅今日",
        "竞价换手率今日",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "竞价匹配价今日",
        "竞价量今日",
        "竞价未匹配量今日",
        "开盘价:不复权今日",
        "成交金额昨日",
        "成交量昨日",
        "成交量前日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
        "个股热度排名昨日",
        "连续涨停天数昨日",
    }
    # If date markers remain after replace_date_markers(), the row came from a
    # different date than requested. This catches the historical replay bug where
    # "1月16日" was resolved by iwencai as the current year.
    date_sensitive_excludes = [r"\[\d{8}"]
    for target, keywords in keyword_mapping.items():
        excludes = date_sensitive_excludes.copy() if target in date_sensitive_targets else []
        if target == "开盘价:不复权今日":
            excludes.append(r"分时")
        source = pick_column(
            columns,
            keywords,
            exclude_patterns=excludes,
        )
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
        "竞价匹配价今日",
        "竞价量今日",
        "竞价未匹配量今日",
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


def compute_factors(df: pd.DataFrame, auction_ratio_mode: str = DEFAULT_AUCTION_RATIO_MODE) -> pd.DataFrame:
    out = df.copy()
    if "竞价未匹配金额" not in out.columns:
        out["竞价未匹配金额"] = pd.Series(index=out.index, dtype="float64")
    auction_amount = pd.to_numeric(out["竞价匹配金额_openapi"], errors="coerce")
    if "成交金额昨日" in out.columns:
        out["竞昨成交比_成交金额口径"] = auction_amount / pd.to_numeric(out["成交金额昨日"], errors="coerce")
    else:
        out["竞昨成交比_成交金额口径"] = pd.Series(index=out.index, dtype="float64")
    required_estimated_columns = {"开盘价:不复权今日", "竞价涨幅今日", "成交量昨日"}
    if required_estimated_columns.issubset(out.columns):
        today_open = pd.to_numeric(out["开盘价:不复权今日"], errors="coerce")
        auction_change = pd.to_numeric(out["竞价涨幅今日"], errors="coerce") / 100
        yesterday_volume = pd.to_numeric(out["成交量昨日"], errors="coerce")
        out["昨收估算"] = today_open / (1 + auction_change)
        out["昨日成交额估算"] = out["昨收估算"] * yesterday_volume
        out["竞昨成交比估算"] = auction_amount / out["昨日成交额估算"].replace(0, pd.NA)
    else:
        out["竞昨成交比估算"] = pd.Series(index=out.index, dtype="float64")
    if auction_ratio_mode == "amount":
        out["竞昨成交比"] = out["竞昨成交比_成交金额口径"]
    elif auction_ratio_mode == "estimated":
        out["竞昨成交比"] = out["竞昨成交比估算"]
    else:
        raise ValueError(f"未知竞昨成交比口径: {auction_ratio_mode}")
    out["竞昨成交比口径"] = auction_ratio_mode
    out["昨日前日成交量比"] = pd.to_numeric(out["成交量昨日"], errors="coerce") / pd.to_numeric(out["成交量前日"], errors="coerce")
    out["竞价未匹配占比"] = pd.to_numeric(out["竞价未匹配金额"], errors="coerce") / pd.to_numeric(
        out["竞价匹配金额_openapi"], errors="coerce"
    )
    return out


def apply_strategy(
    df: pd.DataFrame,
    snapshot: dict[str, Any],
    trade_date: pd.Timestamp,
    sector_source: str,
    min_auction_ratio: float | None = MIN_AUCTION_TO_YESTERDAY_RATIO,
    top_n: int = TOP_N,
    industry_filter_enabled: bool = DEFAULT_INDUSTRY_FILTER_ENABLED,
    min_unmatched_ratio: float | None = DEFAULT_MIN_UNMATCHED_RATIO,
    dynamic_top_n_enabled: bool = DYNAMIC_TOP_N_ENABLED,
    min_auction_change: float | None = MIN_AUCTION_CHANGE,
    max_auction_change: float | None = MAX_AUCTION_CHANGE,
    execution_advice_enabled: bool = EXECUTION_ADVICE_ENABLED,
    min_market20_high_low_diff: float | None = MIN_MARKET_20_HIGH_LOW_DIFF,
    min_market60_high_low_diff: float | None = MIN_MARKET_60_HIGH_LOW_DIFF,
    min_market120_high_low_diff: float | None = MIN_MARKET_120_HIGH_LOW_DIFF,
    loss_cooldown_status: dict[str, Any] | None = None,
    prev_body_min: float = PREV_BODY_MIN,
    auction_ratio_mode: str = DEFAULT_AUCTION_RATIO_MODE,
    dynamic_auction_ratio_enabled: bool = DYNAMIC_AUCTION_RATIO_ENABLED,
    volume_shape_risk_filter_enabled: bool = VOLUME_SHAPE_RISK_FILTER_ENABLED,
    early_weak_continuation_risk_filter_enabled: bool = EARLY_WEAK_CONTINUATION_RISK_FILTER_ENABLED,
    min_prevday_volume_ratio: float | None = YESTERDAY_PREV_VOLUME_RATIO_MIN,
    prevday_volume_ratio_filter_enabled: bool = YESTERDAY_PREV_VOLUME_RATIO_FILTER_ENABLED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if top_n < 1:
        raise ValueError("top_n 必须大于等于 1")
    effective_top_n = resolve_effective_top_n(snapshot, top_n, dynamic_top_n_enabled)
    loss_cooldown_status = loss_cooldown_status or {
        "启用": False,
        "阈值": LOSS_COOLDOWN_CONSECUTIVE_LOSSES,
        "冷却天数": LOSS_COOLDOWN_DAYS,
        "交易记录文件": str(TRADE_RESULT_CSV),
        "最近亏损数": 0,
        "触发日期": None,
        "冷却剩余交易日": 0,
        "触发": False,
        "说明": "连续亏损冷却建议未检查",
    }
    auction_ratio_layer, effective_auction_ratio = resolve_dynamic_auction_ratio_threshold(
        snapshot,
        min_auction_ratio,
        dynamic_auction_ratio_enabled,
    )
    status = {
        "交易日期": trade_date.strftime("%Y-%m-%d"),
        "市场快照日期": snapshot.get("日期"),
        "市场20日高低差": snapshot.get("市场20日高低差"),
        "市场60日高低差": snapshot.get("市场60日高低差"),
        "市场120日高低差": snapshot.get("市场120日高低差"),
        "开仓开关": snapshot.get("开仓开关"),
        "行业强度口径": sector_source,
        "行业过滤启用": industry_filter_enabled,
        "行业涨幅阈值": INDUSTRY_CHANGE_MIN,
        "前日实体阈值": prev_body_min,
        "竞昨成交比口径": auction_ratio_mode,
        "竞昨成交比阈值": effective_auction_ratio,
        "竞昨成交比基础阈值": min_auction_ratio,
        "动态竞昨阈值启用": dynamic_auction_ratio_enabled,
        "动态竞昨市场分层": auction_ratio_layer,
        "动态竞昨强市场20阈值": DYNAMIC_AUCTION_RATIO_STRONG_MARKET20_DIFF,
        "动态竞昨弱市场20阈值": DYNAMIC_AUCTION_RATIO_WEAK_MARKET20_DIFF,
        "动态竞昨强市场阈值": DYNAMIC_AUCTION_RATIO_STRONG,
        "动态竞昨中市场阈值": min_auction_ratio,
        "动态竞昨弱市场阈值": DYNAMIC_AUCTION_RATIO_WEAK,
        "竞价涨幅下限": min_auction_change,
        "竞价涨幅上限": max_auction_change,
        "市场20日高低差阈值": min_market20_high_low_diff,
        "市场60日高低差阈值": min_market60_high_low_diff,
        "市场120日高低差阈值": min_market120_high_low_diff,
        "连续亏损冷却启用": loss_cooldown_status.get("启用"),
        "连续亏损阈值": loss_cooldown_status.get("阈值"),
        "连续亏损冷却天数": loss_cooldown_status.get("冷却天数"),
        "连续亏损交易记录文件": loss_cooldown_status.get("交易记录文件"),
        "连续亏损最近亏损数": loss_cooldown_status.get("最近亏损数"),
        "连续亏损触发日期": loss_cooldown_status.get("触发日期"),
        "连续亏损冷却剩余交易日": loss_cooldown_status.get("冷却剩余交易日"),
        "连续亏损冷却触发": loss_cooldown_status.get("触发"),
        "连续亏损冷却说明": loss_cooldown_status.get("说明"),
        "未匹配占比阈值": min_unmatched_ratio,
        "昨日前日成交量比下限": min_prevday_volume_ratio,
        "昨日前日成交量比过滤启用": prevday_volume_ratio_filter_enabled,
        "仓位规则": POSITION_WEIGHT_POLICY_NAME,
        "仓位市场分层": resolve_market_layer(snapshot),
        "配置最大入选数": top_n,
        "最大入选数": effective_top_n,
        "动态持仓启用": dynamic_top_n_enabled,
        "动态持仓强市场阈值": DYNAMIC_TOP_N_STRONG_MARKET_DIFF,
        "动态持仓中市场阈值": DYNAMIC_TOP_N_MIDDLE_MARKET_DIFF,
        "盘后弱承接风控阈值": ENTRY_DAY_LOW_FROM_OPEN_RISK_EXIT,
        "挂单建议启用": execution_advice_enabled,
        "昨日前日量比风险阈值": YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD,
        "量价结构风险过滤启用": volume_shape_risk_filter_enabled,
        "量价结构风险规则": describe_volume_shape_risk_rule(),
        "早期弱承接风险过滤启用": early_weak_continuation_risk_filter_enabled,
        "早期弱承接风险规则": describe_early_weak_continuation_risk_rule(),
        "原始候选数": int(len(df)),
        "昨日前日成交量比过滤后": 0,
        "昨日前日成交量比过滤数": 0,
        "早期弱承接风险过滤后": 0,
        "早期弱承接风险过滤数": 0,
    }

    market_filter_specs = [
        ("市场20日高低差", min_market20_high_low_diff),
        ("市场60日高低差", min_market60_high_low_diff),
        ("市场120日高低差", min_market120_high_low_diff),
    ]
    market_values: dict[str, float] = {}
    market_fail_reasons: list[str] = []
    for field, threshold in market_filter_specs:
        value = pd.to_numeric(pd.Series([snapshot.get(field)]), errors="coerce").iloc[0]
        if pd.notna(value):
            market_values[field] = float(value)
        if threshold is None:
            continue
        if pd.isna(value):
            raise RuntimeError(f"缺少{field}，无法执行市场环境过滤")
        if float(value) < threshold:
            market_fail_reasons.append(f"{field} {float(value):.0f} < {threshold:g}")
    if market_fail_reasons:
        status["开仓开关"] = "不通过"
        status["市场宽度过滤后"] = 0
        status["市场60过滤后"] = 0
        status["金额过滤后"] = 0
        status["竞昨过滤后"] = 0
        status["竞价涨幅过滤后"] = 0
        status["实体过滤后"] = 0
        status["量价结构风险过滤后"] = 0
        status["量价结构风险过滤数"] = 0
        status["早期弱承接风险过滤后"] = 0
        status["早期弱承接风险过滤数"] = 0
        status["昨日前日成交量比过滤后"] = 0
        status["昨日前日成交量比过滤数"] = 0
        status["行业过滤后"] = 0
        status["未匹配过滤后"] = 0
        status["最终候选数"] = 0
        status["入选数"] = 0
        status["入选量比风险数"] = 0
        status["结果说明"] = "；".join(market_fail_reasons) + "，今日空仓"
        empty = df.head(0).copy()
        for field, value in market_values.items():
            empty[field] = value
        return empty, empty.copy(), status
    status["开仓开关"] = "通过"
    status["市场宽度过滤后"] = int(len(df))
    status["市场60过滤后"] = int(len(df))

    filtered = df.copy()
    for field, value in market_values.items():
        filtered[field] = value
    filtered = filtered[filtered["竞价匹配金额_openapi"] >= MIN_AUCTION_AMOUNT].copy()
    status["金额过滤后"] = int(len(filtered))
    if effective_auction_ratio is not None:
        if "竞昨成交比" not in filtered.columns:
            raise RuntimeError("缺少竞昨成交比，无法执行竞昨成交比过滤")
        auction_ratio = pd.to_numeric(filtered["竞昨成交比"], errors="coerce")
        filtered = filtered[auction_ratio >= effective_auction_ratio].copy()
    status["竞昨过滤后"] = int(len(filtered))
    if min_auction_change is not None or max_auction_change is not None:
        if "竞价涨幅今日" not in filtered.columns:
            raise RuntimeError("缺少竞价涨幅今日，无法执行竞价涨幅过滤")
        auction_change = pd.to_numeric(filtered["竞价涨幅今日"], errors="coerce")
        auction_change_mask = auction_change.notna()
        if min_auction_change is not None:
            auction_change_mask &= auction_change >= min_auction_change
        if max_auction_change is not None:
            auction_change_mask &= auction_change <= max_auction_change
        filtered = filtered[auction_change_mask].copy()
    status["竞价涨幅过滤后"] = int(len(filtered))
    yesterday_body = pd.to_numeric(filtered["实体涨跌幅昨日"], errors="coerce")
    prev_body = pd.to_numeric(filtered["实体涨跌幅前日"], errors="coerce")
    filtered = filtered[(yesterday_body < prev_body) & (prev_body >= prev_body_min)].copy()
    status["实体过滤后"] = int(len(filtered))
    if not filtered.empty:
        required_volume_shape_columns = [
            "昨日前日成交量比",
            "实体涨跌幅昨日",
            "实体涨跌幅前日",
            "市场20日高低差",
        ]
        missing_volume_shape_columns = [
            column for column in required_volume_shape_columns if column not in filtered.columns
        ]
        if missing_volume_shape_columns:
            raise RuntimeError(
                "缺少量价结构风险过滤字段: " + "、".join(missing_volume_shape_columns)
            )
        shape_risk_records = [build_volume_shape_risk(row) for _, row in filtered.iterrows()]
        for column in ["量价结构风险", "量价结构提示"]:
            filtered[column] = [record[column] for record in shape_risk_records]
        if volume_shape_risk_filter_enabled:
            shape_risk_mask = filtered["量价结构风险"].eq("剔除")
            status["量价结构风险过滤数"] = int(shape_risk_mask.sum())
            filtered = filtered[~shape_risk_mask].copy()
        else:
            status["量价结构风险过滤数"] = 0
    else:
        status["量价结构风险过滤数"] = 0
    status["量价结构风险过滤后"] = int(len(filtered))
    if not filtered.empty:
        required_early_risk_columns = [
            "市场20日高低差",
            "市场120日高低差",
            "实体涨跌幅昨日",
            "竞价涨幅今日",
        ]
        missing_early_risk_columns = [
            column for column in required_early_risk_columns if column not in filtered.columns
        ]
        if missing_early_risk_columns:
            raise RuntimeError(
                "缺少早期弱承接风险过滤字段: " + "、".join(missing_early_risk_columns)
            )
        early_risk_records = [build_early_weak_continuation_risk(row) for _, row in filtered.iterrows()]
        for column in ["早期弱承接风险", "早期弱承接提示"]:
            filtered[column] = [record[column] for record in early_risk_records]
        if early_weak_continuation_risk_filter_enabled:
            early_risk_mask = filtered["早期弱承接风险"].eq("剔除")
            status["早期弱承接风险过滤数"] = int(early_risk_mask.sum())
            filtered = filtered[~early_risk_mask].copy()
        else:
            status["早期弱承接风险过滤数"] = 0
    else:
        status["早期弱承接风险过滤数"] = 0
    status["早期弱承接风险过滤后"] = int(len(filtered))
    if industry_filter_enabled:
        if "申万一级行业涨跌幅" not in filtered.columns:
            raise RuntimeError("缺少申万一级行业涨跌幅，无法执行行业联动过滤")
        filtered = filtered[pd.to_numeric(filtered["申万一级行业涨跌幅"], errors="coerce") > INDUSTRY_CHANGE_MIN].copy()
    status["行业过滤后"] = int(len(filtered))
    if prevday_volume_ratio_filter_enabled and min_prevday_volume_ratio is not None:
        if "昨日前日成交量比" not in filtered.columns:
            raise RuntimeError("缺少昨日前日成交量比，无法执行昨日前日成交量比过滤")
        prevday_volume_ratio = pd.to_numeric(filtered["昨日前日成交量比"], errors="coerce")
        before_prevday_volume_filter = int(len(filtered))
        filtered = filtered[prevday_volume_ratio >= min_prevday_volume_ratio].copy()
        status["昨日前日成交量比过滤数"] = before_prevday_volume_filter - int(len(filtered))
    else:
        status["昨日前日成交量比过滤数"] = 0
    status["昨日前日成交量比过滤后"] = int(len(filtered))
    if min_unmatched_ratio is not None:
        if "竞价未匹配占比" not in filtered.columns:
            raise RuntimeError("缺少竞价未匹配占比，无法执行未匹配占比过滤")
        unmatched_ratio = pd.to_numeric(filtered["竞价未匹配占比"], errors="coerce")
        filtered = filtered[unmatched_ratio >= min_unmatched_ratio].copy()
    status["未匹配过滤后"] = int(len(filtered))
    filtered = filtered.sort_values(
        ["竞价涨幅今日", "竞价未匹配占比", "竞昨成交比", "个股热度排名昨日", "基础代码"],
        ascending=[True, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    filtered["排序名次"] = range(1, len(filtered) + 1)
    if not filtered.empty:
        risk_records = [build_volume_ratio_risk(row) for _, row in filtered.iterrows()]
        for column in ["昨日前日量比风险", "昨日前日量比提示"]:
            filtered[column] = [record[column] for record in risk_records]
    selected = filtered.head(effective_top_n).copy()
    if not selected.empty:
        weight_layer, weights = resolve_position_weights_for_selection(snapshot, selected)
        status["仓位市场分层"] = weight_layer
        selected["建议动作"] = "开盘按建议权重买入"
        selected["建议权重"] = [round(weight, 4) for weight in weights]
        selected["仓位规则"] = POSITION_WEIGHT_POLICY_NAME
        selected["仓位市场分层"] = weight_layer
        if execution_advice_enabled:
            advice_records = [build_execution_advice(row, weight_layer) for _, row in selected.iterrows()]
            for column in ["挂单建议", "建议挂单溢价", "挂单上限溢价", "挂单建议理由"]:
                selected[column] = [record[column] for record in advice_records]
    status["最终候选数"] = int(len(filtered))
    status["入选数"] = int(len(selected))
    status["入选量比风险数"] = (
        int(selected["昨日前日量比风险"].eq("谨慎").sum())
        if "昨日前日量比风险" in selected.columns
        else 0
    )
    result_note = "市场开关通过" if not selected.empty else "市场开关通过，但无符合条件标的"
    if loss_cooldown_status.get("触发"):
        result_note += f"；连续亏损冷却建议触发，仅提示不强制空仓：{loss_cooldown_status.get('说明')}"
    status["结果说明"] = result_note
    return filtered, selected, status


def format_ratio_threshold(value: Any) -> str:
    if value is None:
        return "关闭"
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "-"
    return f">={float(number):.4f}"


def format_change_filter(min_value: Any, max_value: Any) -> str:
    min_number = pd.to_numeric(min_value, errors="coerce")
    max_number = pd.to_numeric(max_value, errors="coerce")
    if pd.isna(min_number) and pd.isna(max_number):
        return "关闭"
    if pd.isna(min_number):
        return f"<={float(max_number):g}"
    if pd.isna(max_number):
        return f">={float(min_number):g}"
    return f"{float(min_number):g} 到 {float(max_number):g}"


def format_market_threshold(value: Any) -> str:
    if value is None:
        return "关闭"
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "关闭"
    return f">={float(number):g}"


def format_percent(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "-"
    return f"{float(number) * 100:.1f}%"


def build_risk_exit_note(status: dict[str, Any]) -> str:
    threshold = format_percent(status.get("盘后弱承接风控阈值", ENTRY_DAY_LOW_FROM_OPEN_RISK_EXIT))
    return f"盘后弱承接风控: 买入当日最低价较开盘 <= {threshold} 时，次日开盘优先风控卖；否则按 T1 收盘卖，涨停续持。"


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
        "竞价匹配价今日",
        "竞价量今日",
        "竞价未匹配量今日",
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
        "昨收估算",
        "昨日成交额估算",
        "竞昨成交比估算",
        "竞昨成交比_成交金额口径",
        "竞昨成交比",
        "竞昨成交比口径",
        "昨日前日成交量比",
        "昨日前日量比风险",
        "昨日前日量比提示",
        "量价结构风险",
        "量价结构提示",
        "早期弱承接风险",
        "早期弱承接提示",
        "市场20日高低差",
        "市场60日高低差",
        "市场120日高低差",
    ]
    keep_candidate = [column for column in candidate_columns if column in export_filtered.columns]
    keep_selected = [
        column
        for column in candidate_columns
        + ["建议动作", "建议权重", "挂单建议", "建议挂单溢价", "挂单上限溢价", "挂单建议理由", "仓位规则", "仓位市场分层"]
        if column in export_selected.columns
    ]

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
        f"- 市场20日高低差: `{status.get('市场20日高低差')}` / 阈值 `{format_market_threshold(status.get('市场20日高低差阈值'))}`",
        f"- 市场60日高低差: `{status.get('市场60日高低差')}` / 阈值 `{format_market_threshold(status.get('市场60日高低差阈值'))}`",
        f"- 市场120日高低差: `{status.get('市场120日高低差')}` / 阈值 `{format_market_threshold(status.get('市场120日高低差阈值'))}`",
        f"- 开仓开关: `{status.get('开仓开关')}`",
        f"- 行业强度口径: `{status.get('行业强度口径')}`",
        f"- 行业过滤: `{'启用' if status.get('行业过滤启用') else '关闭'}`",
        f"- 行业涨幅阈值: `>{status.get('行业涨幅阈值')}`",
        f"- 前日实体阈值: `>={status.get('前日实体阈值')}`",
        f"- 竞昨成交比口径: `{status.get('竞昨成交比口径')}`",
        f"- 竞昨成交比阈值: `{format_ratio_threshold(status.get('竞昨成交比阈值'))}`",
        f"- 动态竞昨阈值: `{'启用' if status.get('动态竞昨阈值启用') else '关闭'}` / `{status.get('动态竞昨市场分层')}`市场 / 强`>={status.get('动态竞昨强市场20阈值')}`用`{format_ratio_threshold(status.get('动态竞昨强市场阈值'))}` / 中用`{format_ratio_threshold(status.get('动态竞昨中市场阈值'))}` / 弱`<{status.get('动态竞昨弱市场20阈值')}`用`{format_ratio_threshold(status.get('动态竞昨弱市场阈值'))}`",
        f"- 竞价涨幅过滤: `{format_change_filter(status.get('竞价涨幅下限'), status.get('竞价涨幅上限'))}`",
        f"- 未匹配占比阈值: `{format_ratio_threshold(status.get('未匹配占比阈值'))}`",
        f"- 连续亏损冷却建议: `{'建议冷却' if status.get('连续亏损冷却触发') else ('启用' if status.get('连续亏损冷却启用') else '关闭')}` / 最近亏损 `{status.get('连续亏损最近亏损数')}` / 剩余 `{status.get('连续亏损冷却剩余交易日')}` 个交易日 / `仅提示不强制空仓`",
        f"- 连续亏损建议说明: {status.get('连续亏损冷却说明')}",
        f"- 昨日前日成交量比过滤: `{'启用' if status.get('昨日前日成交量比过滤启用') else '关闭'}` / 下限 `{format_ratio_threshold(status.get('昨日前日成交量比下限'))}` / 过滤后 `{status.get('昨日前日成交量比过滤后', 0)}` / 剔除 `{status.get('昨日前日成交量比过滤数', 0)}`",
        f"- 昨日前日量比风险提示: `>{status.get('昨日前日量比风险阈值', YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD)} 标记谨慎，不默认剔除` / 入选风险数 `{status.get('入选量比风险数', 0)}`",
        f"- 量价结构风险过滤: `{'启用' if status.get('量价结构风险过滤启用') else '关闭'}` / 剔除 `{status.get('量价结构风险过滤数', 0)}` / 规则 `{status.get('量价结构风险规则')}`",
        f"- 早期弱承接过滤: `{'启用' if status.get('早期弱承接风险过滤启用') else '关闭'}` / 剔除 `{status.get('早期弱承接风险过滤数', 0)}` / 规则 `{status.get('早期弱承接风险规则')}`",
        f"- 仓位规则: `{status.get('仓位规则')}` / `{status.get('仓位市场分层')}市场`",
        f"- 挂单建议: `{'启用' if status.get('挂单建议启用') else '关闭'}`",
        f"- 动态持仓: `{'启用' if status.get('动态持仓启用') else '关闭'}`",
        f"- 动态持仓档位: `>={status.get('动态持仓强市场阈值')}取TOP3，>={status.get('动态持仓中市场阈值')}取TOP2，否则TOP1`",
        f"- 配置最大入选数: `{status.get('配置最大入选数', TOP_N)}`",
        f"- 最大入选数: `{status.get('最大入选数', TOP_N)}`",
        f"- {build_risk_exit_note(status)}",
        f"- 原始候选数: `{status.get('原始候选数')}`",
        f"- 市场宽度过滤后: `{status.get('市场宽度过滤后', 0)}`",
        f"- 金额过滤后: `{status.get('金额过滤后', 0)}`",
        f"- 竞昨过滤后: `{status.get('竞昨过滤后', 0)}`",
        f"- 竞价涨幅过滤后: `{status.get('竞价涨幅过滤后', 0)}`",
        f"- 实体过滤后: `{status.get('实体过滤后', 0)}`",
        f"- 量价结构风险过滤后: `{status.get('量价结构风险过滤后', 0)}`",
        f"- 行业过滤后: `{status.get('行业过滤后', 0)}`",
        f"- 未匹配过滤后: `{status.get('未匹配过滤后', 0)}`",
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
                "| 排名 | 股票代码 | 股票简称 | 一级行业 | 行业涨幅 | 行业涨幅排名 | 竞价金额 | 未匹配占比 | 竞昨成交比 | 昨前量比 | 量比风险 | 量价风险 | 早期弱承接 | 热度排名昨日 | 建议权重 | 建议动作 | 挂单建议 | 建议溢价 | 上限溢价 | 挂单理由 |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for _, row in export_selected.iterrows():
            lines.append(
                f"| {int(row['排序名次'])} | {row['股票代码']} | {row['股票简称']} | "
                f"{row.get('申万一级行业', '')} | {row.get('申万一级行业涨跌幅', float('nan')):.4f} | "
                f"{row.get('申万一级行业涨跌幅排名', float('nan')):.0f} | "
                f"{row['竞价匹配金额_openapi']:.0f} | {row.get('竞价未匹配占比', float('nan')):.4f} | {row['竞昨成交比']:.4f} | "
                f"{row.get('昨日前日成交量比', float('nan')):.2f} | {row.get('昨日前日量比风险', '-')} | "
                f"{row.get('量价结构风险', '-')} | {row.get('早期弱承接风险', '-')} | "
                f"{row['个股热度排名昨日']:.0f} | {row.get('建议权重', float('nan')):.4f} | {row['建议动作']} | "
                f"{row.get('挂单建议', '-')} | {format_percent(row.get('建议挂单溢价'))} | {format_percent(row.get('挂单上限溢价'))} | "
                f"{row.get('挂单建议理由', '')} |"
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
    top_n = int(status.get("最大入选数") or TOP_N)
    lines = [
        f"交易日期: {trade_date.strftime('%Y-%m-%d')}",
        f"市场开关: {status.get('开仓开关')} / 市场20日={status.get('市场20日高低差')} / 市场60日={status.get('市场60日高低差')} / 市场120日={status.get('市场120日高低差')}",
        f"行业口径: {status.get('行业强度口径')}",
        f"行业过滤: {'启用' if status.get('行业过滤启用') else '关闭'}",
        f"市场宽度阈值: 20日{format_market_threshold(status.get('市场20日高低差阈值'))} / 60日{format_market_threshold(status.get('市场60日高低差阈值'))} / 120日{format_market_threshold(status.get('市场120日高低差阈值'))}",
        f"竞昨成交比阈值: {format_ratio_threshold(status.get('竞昨成交比阈值'))}",
        f"动态竞昨阈值: {'启用' if status.get('动态竞昨阈值启用') else '关闭'} / {status.get('动态竞昨市场分层')}市场 / 强>={status.get('动态竞昨强市场20阈值')}用{format_ratio_threshold(status.get('动态竞昨强市场阈值'))} / 中用{format_ratio_threshold(status.get('动态竞昨中市场阈值'))} / 弱<{status.get('动态竞昨弱市场20阈值')}用{format_ratio_threshold(status.get('动态竞昨弱市场阈值'))}",
        f"竞价涨幅过滤: {format_change_filter(status.get('竞价涨幅下限'), status.get('竞价涨幅上限'))}",
        f"未匹配占比阈值: {format_ratio_threshold(status.get('未匹配占比阈值'))}",
        f"昨日前日成交量比过滤: {'启用' if status.get('昨日前日成交量比过滤启用') else '关闭'} / 下限{format_ratio_threshold(status.get('昨日前日成交量比下限'))} / 过滤后{status.get('昨日前日成交量比过滤后', 0)} / 剔除{status.get('昨日前日成交量比过滤数', 0)}",
        f"连续亏损冷却建议: {'建议冷却' if status.get('连续亏损冷却触发') else ('启用' if status.get('连续亏损冷却启用') else '关闭')} / 最近亏损{status.get('连续亏损最近亏损数')} / 剩余{status.get('连续亏损冷却剩余交易日')}日 / 仅提示不强制空仓",
        f"昨日前日量比风险: >{status.get('昨日前日量比风险阈值', YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD)} 标记谨慎，不默认剔除 / 入选风险数{status.get('入选量比风险数', 0)}",
        f"量价结构风险过滤: {'启用' if status.get('量价结构风险过滤启用') else '关闭'} / 剔除{status.get('量价结构风险过滤数', 0)}",
        f"早期弱承接过滤: {'启用' if status.get('早期弱承接风险过滤启用') else '关闭'} / 剔除{status.get('早期弱承接风险过滤数', 0)}",
        f"仓位规则: {status.get('仓位规则')} / {status.get('仓位市场分层')}市场",
        f"挂单建议: {'启用' if status.get('挂单建议启用') else '关闭'}",
        f"动态持仓: {'启用' if status.get('动态持仓启用') else '关闭'} / 配置TOP{status.get('配置最大入选数', TOP_N)} -> 今日TOP{status.get('最大入选数', TOP_N)}",
        build_risk_exit_note(status),
        f"过滤: 原始{status.get('原始候选数', 0)} -> 市场宽度{status.get('市场宽度过滤后', 0)} -> 金额{status.get('金额过滤后', 0)} -> 竞昨{status.get('竞昨过滤后', 0)} -> 竞价涨幅{status.get('竞价涨幅过滤后', 0)} -> 实体{status.get('实体过滤后', 0)} -> 量价结构{status.get('量价结构风险过滤后', 0)} -> 行业{status.get('行业过滤后', 0)} -> 未匹配{status.get('未匹配过滤后', 0)} -> 入选{status.get('入选数', 0)}",
        f"说明: {status.get('结果说明')}",
        "",
    ]

    if selected_df.empty:
        lines.append("今日无可操作标的")
        return "\n".join(lines)

    lines.append(f"今日前{top_n}标的:")
    for _, row in selected_df.iterrows():
        lines.extend(
            [
                f"{int(row['排序名次'])}. {row['股票简称']}（{row['股票代码']}）",
                f"行业: {row.get('申万一级行业', '-')}",
                f"建议权重: {format_push_number(row.get('建议权重'), 4)}",
                f"竞价金额: {format_push_number(row.get('竞价匹配金额_openapi'), 0)}",
                f"未匹配占比: {format_push_number(row.get('竞价未匹配占比'), 4)}",
                f"竞昨成交比: {format_push_number(row.get('竞昨成交比'), 4)}",
                f"昨日前日量比: {format_push_number(row.get('昨日前日成交量比'), 2)} / 风险: {row.get('昨日前日量比风险', '-')}",
                f"量价结构风险: {row.get('量价结构风险', '-')}",
                f"早期弱承接风险: {row.get('早期弱承接风险', '-')}",
                f"昨日热度排名: {format_push_number(row.get('个股热度排名昨日'), 0)}",
                f"挂单建议: {row.get('挂单建议', '-')} / 建议溢价{format_percent(row.get('建议挂单溢价'))} / 上限{format_percent(row.get('挂单上限溢价'))}",
                f"挂单理由: {row.get('挂单建议理由', '-')}",
                f"量比提示: {row.get('昨日前日量比提示', '-') or '-'}",
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
    market60_filter_required = not args.no_market60_filter and args.min_market60_high_low_diff is not None
    market120_filter_required = not args.no_market120_filter and args.min_market120_high_low_diff is not None
    snapshot = read_market_snapshot(
        today_ts,
        historical_replay,
        required_snapshot_date,
        market60_filter_required,
        market120_filter_required,
    )
    logging.info(
        "市场快照 date=%s diff20=%s diff60=%s diff120=%s switch=%s",
        snapshot.get("日期"),
        snapshot.get("市场20日高低差"),
        snapshot.get("市场60日高低差"),
        snapshot.get("市场120日高低差"),
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
    raw_frames: dict[str, pd.DataFrame] = {}
    standardized_frames: dict[str, pd.DataFrame] = {}
    for label, question in queries.items():
        frame = query_wencai(question, cookies)
        logging.info("问财查询完成 label=%s rows=%s", label, len(frame))
        raw_frames[label] = frame.copy()
        standardized = standardize_frame(frame, date_map)
        standardized_frames[label] = standardized.copy()
        frames.append(standardized)

    merged = merge_frames(frames)
    if merged.empty:
        raise RuntimeError("问财返回为空，无法生成操作清单")

    merged = compute_factors(merged, args.auction_ratio_mode)
    merged, sector_df, sector_source = attach_sector_context(merged, today_ts, historical_replay)
    min_auction_ratio = None if args.no_auction_ratio_filter else args.min_auction_ratio
    min_auction_change = None if args.no_auction_change_filter else args.min_auction_change
    max_auction_change = None if args.no_auction_change_filter else args.max_auction_change
    min_market20_high_low_diff = None if args.no_market20_filter else args.min_market20_high_low_diff
    min_market60_high_low_diff = None if args.no_market60_filter else args.min_market60_high_low_diff
    min_market120_high_low_diff = None if args.no_market120_filter else args.min_market120_high_low_diff
    min_prevday_volume_ratio = None if args.no_prevday_volume_ratio_filter else args.min_prevday_volume_ratio
    loss_cooldown_status = build_loss_cooldown_status(
        today_ts,
        args.trade_result_file,
        not args.no_loss_cooldown,
        args.loss_cooldown_consecutive_losses,
        args.loss_cooldown_days,
    )
    logging.info("连续亏损冷却状态: %s", loss_cooldown_status)
    filtered, selected, status = apply_strategy(
        merged,
        snapshot,
        today_ts,
        sector_source,
        min_auction_ratio,
        args.top_n,
        args.industry_filter,
        args.min_unmatched_ratio,
        not args.fixed_top_n,
        min_auction_change,
        max_auction_change,
        not args.no_execution_advice,
        min_market20_high_low_diff,
        min_market60_high_low_diff,
        min_market120_high_low_diff,
        loss_cooldown_status,
        args.prev_body_min,
        args.auction_ratio_mode,
        args.dynamic_auction_ratio,
        not args.no_volume_shape_risk_filter,
        not args.no_early_weak_continuation_filter,
        min_prevday_volume_ratio,
        not args.no_prevday_volume_ratio_filter,
    )
    export_outputs(today_ts, filtered, selected, status, queries)
    archive_dir = archive_daily_data(
        today_ts,
        prev_ts,
        prev2_ts,
        raw_frames,
        standardized_frames,
        sector_df,
        merged,
        filtered,
        selected,
        snapshot,
        status,
        queries,
        date_map,
        sector_source,
        historical_replay,
    )
    should_push = not args.no_push and (not historical_replay or args.push)
    logging.info("推送判断 should_push=%s no_push=%s push=%s", should_push, args.no_push, args.push)
    if should_push:
        push_title = f"{today_ts.strftime('%Y-%m-%d')} 竞价爬升操作清单"
        push_content = build_pushplus_content(today_ts, selected, status)
        send_pushplus(push_title, push_content)
    logging.info(
        "运行成功 trade_date=%s raw=%s market_width=%s cooldown=%s amount=%s auction_ratio=%s auction_change=%s body=%s volume_shape=%s industry=%s unmatched=%s selected=%s",
        status.get("交易日期"),
        status.get("原始候选数"),
        status.get("市场宽度过滤后"),
        status.get("连续亏损冷却触发"),
        status.get("金额过滤后"),
        status.get("竞昨过滤后"),
        status.get("竞价涨幅过滤后"),
        status.get("实体过滤后"),
        status.get("量价结构风险过滤后"),
        status.get("行业过滤后"),
        status.get("未匹配过滤后"),
        status.get("入选数"),
    )

    print(f"交易日期: {status['交易日期']}")
    print(
        f"市场开关: {status['开仓开关']} / 市场20日高低差={status['市场20日高低差']} / "
        f"阈值{format_market_threshold(status.get('市场20日高低差阈值'))} / "
        f"市场60日高低差={status.get('市场60日高低差')} / 阈值{format_market_threshold(status.get('市场60日高低差阈值'))} / "
        f"市场120日高低差={status.get('市场120日高低差')} / 阈值{format_market_threshold(status.get('市场120日高低差阈值'))}"
    )
    print(
        f"行业口径: {status['行业强度口径']} / "
        f"行业过滤={'启用' if status.get('行业过滤启用') else '关闭'} / "
        f"阈值=涨幅>{status['行业涨幅阈值']}"
    )
    print(f"竞昨成交比口径: {status.get('竞昨成交比口径')}")
    print(f"竞昨成交比阈值: {format_ratio_threshold(status.get('竞昨成交比阈值'))}")
    print(f"竞价涨幅过滤: {format_change_filter(status.get('竞价涨幅下限'), status.get('竞价涨幅上限'))}")
    print(f"未匹配占比阈值: {format_ratio_threshold(status.get('未匹配占比阈值'))}")
    print(
        f"昨日前日成交量比过滤: {'启用' if status.get('昨日前日成交量比过滤启用') else '关闭'} / "
        f"下限{format_ratio_threshold(status.get('昨日前日成交量比下限'))} / "
        f"过滤后={status.get('昨日前日成交量比过滤后', 0)} / 剔除={status.get('昨日前日成交量比过滤数', 0)}"
    )
    print(
        f"连续亏损冷却建议: {'建议冷却' if status.get('连续亏损冷却触发') else ('启用' if status.get('连续亏损冷却启用') else '关闭')} / "
        f"最近亏损={status.get('连续亏损最近亏损数')} / 剩余={status.get('连续亏损冷却剩余交易日')}个交易日"
    )
    print("连续亏损冷却建议仅提示不强制空仓")
    print(f"连续亏损建议说明: {status.get('连续亏损冷却说明')}")
    print(
        f"昨日前日量比风险提示: >{status.get('昨日前日量比风险阈值', YESTERDAY_PREV_VOLUME_RATIO_RISK_THRESHOLD)} 标记谨慎，"
        f"不默认剔除 / 入选风险数={status.get('入选量比风险数', 0)}"
    )
    print(
        f"量价结构风险过滤: {'启用' if status.get('量价结构风险过滤启用') else '关闭'} / "
        f"剔除={status.get('量价结构风险过滤数', 0)} / 过滤后={status.get('量价结构风险过滤后', 0)}"
    )
    print(f"仓位规则: {status.get('仓位规则')} / {status.get('仓位市场分层')}市场")
    print(f"挂单建议: {'启用' if status.get('挂单建议启用') else '关闭'}")
    print(
        f"动态持仓: {'启用' if status.get('动态持仓启用') else '关闭'} / "
        f">={status.get('动态持仓强市场阈值')}取TOP3 / "
        f">={status.get('动态持仓中市场阈值')}取TOP2 / 否则TOP1 / "
        f"配置TOP{status.get('配置最大入选数')} -> 今日TOP{status.get('最大入选数')}"
    )
    print(build_risk_exit_note(status))
    print(f"最大入选数: {status['最大入选数']}")
    print(f"原始候选数: {status['原始候选数']}")
    print(f"市场宽度过滤后: {status['市场宽度过滤后']}")
    print(f"金额过滤后: {status['金额过滤后']}")
    print(f"竞昨过滤后: {status['竞昨过滤后']}")
    print(f"竞价涨幅过滤后: {status['竞价涨幅过滤后']}")
    print(f"实体过滤后: {status['实体过滤后']}")
    print(f"行业过滤后: {status['行业过滤后']}")
    print(f"未匹配过滤后: {status['未匹配过滤后']}")
    print(f"最终候选数: {status['最终候选数']}")
    print(f"入选数: {status['入选数']}")
    if selected.empty:
        print("今日无可操作标的")
    else:
        print(f"今日前{status['最大入选数']}标的:")
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
                    "昨日前日成交量比",
                    "昨日前日量比风险",
                    "个股热度排名昨日",
                    "建议权重",
                    "挂单建议",
                    "建议挂单溢价",
                    "挂单上限溢价",
                ]
            ].to_string(index=False)
        )
    print(f"数据归档目录: {archive_dir}")
    print(f"每日采集汇总: {ARCHIVE_SUMMARY_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("生成失败")
        print(f"生成失败: {exc}")
        raise SystemExit(1)
