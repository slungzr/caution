from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import requests
from akshare.stock_feature.stock_a_indicator import headers as LEGU_HEADERS

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "竞价爬升-20240504.csv"
FACTOR_INPUT_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
OUTPUT_STEM = INPUT_CSV.with_suffix("")
CACHE_DIR = BASE_DIR / "cache" / "daily_history"

# 回测规则
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 3
BUY_RANK_COLUMN = "个股热度排名昨日"

# 成本口径: 买卖双边佣金 + 卖出印花税 + 双边滑点
BUY_COMMISSION_RATE = 0.0003
SELL_COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005
OPEN_PRICE_MISMATCH_THRESHOLD = 0.003

# 硬过滤入口。当前默认不启用，先把真实持仓回测跑通。
# 示例:
# BUY_FILTERS = {
#     "竞价涨幅今日": {"min": -2.0, "max": 6.0},
#     "量比": {"min": 1.2},
#     "换手率昨日": {"min": 3.0},
#     "大单动向(ddx值)昨日": {"min": 0.0},
# }
BUY_FILTERS: dict[str, dict[str, Any]] = {}

# 市场情绪过滤。当前先补历史市场宽度数据，不默认启用阈值。
MARKET_BREADTH_FILTER = {
    "enabled": False,
    "min_high20_minus_low20": None,
}

MARKET_BREADTH_SYMBOL = "all"
HISTORY_FETCH_WORKERS = 8
HISTORY_FETCH_RETRIES = 3
SCAN_RANK_UNIVERSE_LIMIT = 4
TRAIN_RATIO = 0.7

EXPORT_CANDIDATE_COLUMNS = [
    "日期",
    "股票代码",
    "股票简称",
    "基础代码",
    "开盘价:不复权今日",
    "收盘价:不复权今日",
    "个股热度排名昨日",
    "竞价涨幅今日",
    "量比",
    "换手率昨日",
    "大单动向(ddx值)昨日",
    "实体涨跌幅昨日",
    "实体涨跌幅前日",
    "成交量昨日",
    "成交量前日",
    "连续涨停天数昨日",
    "市场20日新高数",
    "市场20日新低数",
    "市场20日高低差",
    "市场60日新高数",
    "市场60日新低数",
    "市场60日高低差",
]

DEFAULT_NAMED_CONDITIONS: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    buy_filters: dict[str, dict[str, Any]] | None = None
    named_conditions: tuple[str, ...] = DEFAULT_NAMED_CONDITIONS
    market_filter: dict[str, Any] | None = None
    sort_by: tuple[tuple[str, bool], ...] | None = None


NEW_FACTOR_CONFIG = StrategyConfig(
    name="新因子策略版",
    buy_filters={
        "竞价匹配金额_openapi": {"min": 50_000_000},
    },
    named_conditions=("yesterday_body_lt_prev",),
    market_filter={"min_high20_minus_low20": 0},
)

AUCTION_RATIO_RANK_CONFIG = StrategyConfig(
    name="竞昨比排序版",
    buy_filters={
        "竞价匹配金额_openapi": {"min": 50_000_000},
    },
    named_conditions=("yesterday_body_lt_prev",),
    market_filter={"min_high20_minus_low20": 0},
    sort_by=(
        ("竞昨成交比估算", False),
        (BUY_RANK_COLUMN, True),
        ("基础代码", True),
    ),
)


@dataclass
class Position:
    code: str
    raw_code: str
    name: str
    entry_date: pd.Timestamp
    entry_price_raw: float
    entry_price_exec: float
    allocated_cash: float
    shares: float
    signal_rank: float
    last_close: float


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    encodings = ["gb18030", "gbk", "utf-8-sig", "utf-8"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - 依赖本地文件编码
            last_error = exc
    raise RuntimeError(f"读取 {csv_path} 失败") from last_error


def normalize_stock_code(value: Any) -> str:
    digits = re.findall(r"\d+", str(value))
    if not digits:
        return ""
    return "".join(digits)[-6:].zfill(6)


def to_akshare_symbol(code: str) -> str:
    if code.startswith(("4", "8")):
        return f"bj{code}"
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def parse_legu_date(value: Any) -> pd.Timestamp:
    if isinstance(value, list) and len(value) >= 3:
        return pd.Timestamp(year=int(value[0]), month=int(value[1]), day=int(value[2]))
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return pd.to_datetime(int(value), unit="ms").normalize()
    return pd.to_datetime(value).normalize()


def round_price(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def get_limit_up_ratio(code: str, name: str) -> float:
    upper_name = str(name).upper()
    if "ST" in upper_name:
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("4", "8")):
        return 0.30
    return 0.10


def is_limit_up_close(code: str, name: str, close_price: float, prev_close: float) -> bool:
    if pd.isna(close_price) or pd.isna(prev_close) or prev_close <= 0:
        return False
    limit_up_price = round_price(prev_close * (1 + get_limit_up_ratio(code, name)))
    return abs(float(close_price) - limit_up_price) <= 0.011


def apply_filter_rules(df: pd.DataFrame, filter_rules: dict[str, dict[str, Any]]) -> pd.DataFrame:
    filtered = df.copy()
    for column, rule in filter_rules.items():
        if column not in filtered.columns:
            raise KeyError(f"过滤列不存在: {column}")
        numeric_series = pd.to_numeric(filtered[column], errors="coerce")
        mask = numeric_series.notna()
        if rule.get("min") is not None:
            mask &= numeric_series >= rule["min"]
        if rule.get("max") is not None:
            mask &= numeric_series <= rule["max"]
        if rule.get("allowed") is not None:
            mask &= filtered[column].isin(rule["allowed"])
        filtered = filtered.loc[mask].copy()
    return filtered


def apply_named_conditions(df: pd.DataFrame, named_conditions: tuple[str, ...]) -> pd.DataFrame:
    filtered = df.copy()
    for condition in named_conditions:
        if condition == "yesterday_body_lt_prev":
            filtered = filtered[
                pd.to_numeric(filtered["实体涨跌幅昨日"], errors="coerce")
                < pd.to_numeric(filtered["实体涨跌幅前日"], errors="coerce")
            ].copy()
        elif condition == "yesterday_volume_gt_prev":
            filtered = filtered[
                pd.to_numeric(filtered["成交量昨日"], errors="coerce")
                > pd.to_numeric(filtered["成交量前日"], errors="coerce")
            ].copy()
        elif condition == "not_st":
            filtered = filtered[
                ~filtered["股票简称"].astype(str).str.upper().str.contains("ST", regex=False, na=False)
            ].copy()
        else:
            raise KeyError(f"未知命名过滤条件: {condition}")
    return filtered


def format_filter_rules(filter_rules: dict[str, dict[str, Any]] | None) -> str:
    if not filter_rules:
        return ""
    parts: list[str] = []
    for column, rule in filter_rules.items():
        if rule.get("min") is not None:
            parts.append(f"{column}>={rule['min']}")
        if rule.get("max") is not None:
            parts.append(f"{column}<={rule['max']}")
        if rule.get("allowed") is not None:
            parts.append(f"{column} in {rule['allowed']}")
    return "; ".join(parts)


def format_market_filter(market_filter: dict[str, Any] | None) -> str:
    if not market_filter:
        return ""
    parts: list[str] = []
    if market_filter.get("min_high20_minus_low20") is not None:
        parts.append(f"市场20日高低差>={market_filter['min_high20_minus_low20']}")
    if market_filter.get("min_high60_minus_low60") is not None:
        parts.append(f"市场60日高低差>={market_filter['min_high60_minus_low60']}")
    return "; ".join(parts)


def load_signal_data(csv_path: Path) -> pd.DataFrame:
    df = read_csv_with_fallback(csv_path)
    df["日期"] = pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce")
    df["基础代码"] = df["股票代码"].apply(normalize_stock_code)

    numeric_columns = [
        "开盘价:不复权今日",
        "收盘价:不复权今日",
        "个股热度排名昨日",
        "竞价涨幅今日",
        "量比",
        "换手率昨日",
        "大单动向(ddx值)昨日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
        "成交量昨日",
        "成交量前日",
        "连续涨停天数昨日",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["日期", "基础代码", BUY_RANK_COLUMN, "开盘价:不复权今日"]).copy()
    df = df[df["基础代码"] != ""].copy()
    df = df.sort_values(["日期", BUY_RANK_COLUMN, "基础代码"], kind="stable").reset_index(drop=True)
    return df


def load_factor_signal_data(csv_path: Path) -> pd.DataFrame:
    df = load_signal_data(csv_path)
    factor_columns = [
        "竞价强度",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "竞价换手率_openapi",
        "竞价量比_openapi",
        "昨日涨停_openapi",
        "昨日炸板_openapi",
        "连续涨停天数_openapi",
    ]
    for column in factor_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if {"开盘价:不复权今日", "竞价涨幅今日", "成交量昨日", "竞价匹配金额_openapi"}.issubset(df.columns):
        today_open = pd.to_numeric(df["开盘价:不复权今日"], errors="coerce")
        auction_change = pd.to_numeric(df["竞价涨幅今日"], errors="coerce") / 100
        yesterday_volume = pd.to_numeric(df["成交量昨日"], errors="coerce")
        yesterday_close_est = today_open / (1 + auction_change)
        yesterday_amount_est = yesterday_close_est * yesterday_volume
        df["昨收估算"] = yesterday_close_est
        df["昨日成交额估算"] = yesterday_amount_est
        df["竞昨成交比估算"] = pd.to_numeric(df["竞价匹配金额_openapi"], errors="coerce") / yesterday_amount_est
        df["昨日前日成交量比"] = yesterday_volume / pd.to_numeric(df["成交量前日"], errors="coerce")

    if "竞价强度" in df.columns:
        df = df[df["竞价强度"].notna()].copy()
    return df.reset_index(drop=True)


def restrict_signal_universe(signal_df: pd.DataFrame, rank_limit: int) -> pd.DataFrame:
    restricted_df = signal_df.sort_values(["日期", BUY_RANK_COLUMN, "基础代码"], kind="stable")
    restricted_df = restricted_df.groupby("日期", group_keys=False).head(rank_limit).copy()
    restricted_df.reset_index(drop=True, inplace=True)
    return restricted_df


def split_signal_data_by_ratio(signal_df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(signal_df["日期"].dropna().unique().tolist())
    if len(unique_dates) < 2:
        raise ValueError("交易日数量不足，无法切分训练期和验证期")

    split_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * train_ratio)))
    split_date = pd.Timestamp(unique_dates[split_index])
    train_df = signal_df[signal_df["日期"] < split_date].copy()
    valid_df = signal_df[signal_df["日期"] >= split_date].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("训练期或验证期为空，无法进行样本外验证")
    return train_df, valid_df, split_date


def fetch_market_breadth_history(symbol: str = MARKET_BREADTH_SYMBOL) -> pd.DataFrame:
    url = f"https://www.legulegu.com/stockdata/member-ship/get-high-low-statistics/{symbol}"
    response = requests.get(url, headers=LEGU_HEADERS, timeout=30)
    response.raise_for_status()
    raw_df = pd.DataFrame(response.json())
    breadth_df = raw_df.copy()
    breadth_df["日期"] = breadth_df["date"].apply(parse_legu_date)
    numeric_columns = ["close", "high20", "low20", "high60", "low60", "high120", "low120"]
    for column in numeric_columns:
        breadth_df[column] = pd.to_numeric(breadth_df[column], errors="coerce")
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
    breadth_df = breadth_df[keep_columns].drop_duplicates(subset=["日期"]).sort_values("日期")
    return breadth_df.reset_index(drop=True)


def fetch_latest_market_activity() -> pd.DataFrame:
    activity_df = ak.stock_market_activity_legu().copy()
    return activity_df


def build_candidate_book(signal_df: pd.DataFrame, breadth_df: pd.DataFrame) -> pd.DataFrame:
    return build_candidate_book_with_config(
        signal_df=signal_df,
        breadth_df=breadth_df,
        buy_filters=BUY_FILTERS,
        named_conditions=DEFAULT_NAMED_CONDITIONS,
        market_filter=MARKET_BREADTH_FILTER,
        max_positions=MAX_POSITIONS,
    )


def build_candidate_book_with_config(
    signal_df: pd.DataFrame,
    breadth_df: pd.DataFrame,
    buy_filters: dict[str, dict[str, Any]] | None = None,
    named_conditions: tuple[str, ...] = DEFAULT_NAMED_CONDITIONS,
    market_filter: dict[str, Any] | None = None,
    sort_by: tuple[tuple[str, bool], ...] | None = None,
    max_positions: int = MAX_POSITIONS,
) -> pd.DataFrame:
    candidates = signal_df.copy()
    if not breadth_df.empty:
        candidates = candidates.merge(breadth_df, on="日期", how="left")

    if named_conditions:
        candidates = apply_named_conditions(candidates, named_conditions)

    if buy_filters:
        candidates = apply_filter_rules(candidates, buy_filters)

    if market_filter:
        threshold_20 = market_filter.get("min_high20_minus_low20")
        if threshold_20 is not None and "市场20日高低差" in candidates.columns:
            candidates = candidates[candidates["市场20日高低差"] >= threshold_20].copy()

        threshold_60 = market_filter.get("min_high60_minus_low60")
        if threshold_60 is not None and "市场60日高低差" in candidates.columns:
            candidates = candidates[candidates["市场60日高低差"] >= threshold_60].copy()

    sort_columns = ["日期"]
    ascending = [True]
    if sort_by:
        for column, is_ascending in sort_by:
            if column not in candidates.columns:
                raise KeyError(f"排序列不存在: {column}")
            sort_columns.append(column)
            ascending.append(is_ascending)
        if "基础代码" not in sort_columns:
            sort_columns.append("基础代码")
            ascending.append(True)
    else:
        sort_columns.extend([BUY_RANK_COLUMN, "基础代码"])
        ascending.extend([True, True])

    candidates = candidates.sort_values(sort_columns, ascending=ascending, kind="stable")
    candidate_book = candidates.groupby("日期", group_keys=False).head(max_positions).copy()
    candidate_book.reset_index(drop=True, inplace=True)
    return candidate_book


def fetch_stock_history(code: str) -> tuple[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{code}.csv"
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path, encoding="utf-8-sig")
        if not cached_df.empty:
            cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.normalize()
            for column in ["open", "high", "low", "close", "prev_close"]:
                if column in cached_df.columns:
                    cached_df[column] = pd.to_numeric(cached_df[column], errors="coerce")
            cached_df = cached_df.dropna(subset=["date", "close"]).sort_values("date")
            cached_df = cached_df.drop_duplicates(subset=["date"]).set_index("date")
            return code, cached_df

    symbol = to_akshare_symbol(code)
    last_error: Exception | None = None
    for attempt in range(1, HISTORY_FETCH_RETRIES + 1):
        try:
            history_df = ak.stock_zh_a_daily(symbol=symbol, adjust="")
            if history_df.empty:
                return code, pd.DataFrame()
            history_df = history_df.reset_index(drop=True).copy()
            history_df.columns = [str(column).lower() for column in history_df.columns]
            if "date" not in history_df.columns:
                raise KeyError(f"{code} 缺少 date 列")
            keep_columns = [column for column in ["date", "open", "high", "low", "close"] if column in history_df.columns]
            history_df = history_df[keep_columns].copy()
            history_df["date"] = pd.to_datetime(history_df["date"]).dt.normalize()
            for column in ["open", "high", "low", "close"]:
                if column in history_df.columns:
                    history_df[column] = pd.to_numeric(history_df[column], errors="coerce")
            history_df = history_df.dropna(subset=["date", "close"]).sort_values("date")
            history_df = history_df.drop_duplicates(subset=["date"]).set_index("date")
            history_df["prev_close"] = history_df["close"].shift(1)
            history_df.reset_index().to_csv(cache_path, index=False, encoding="utf-8-sig")
            return code, history_df
        except Exception as exc:  # pragma: no cover - 依赖网络
            last_error = exc
            time.sleep(attempt)
    raise RuntimeError(f"拉取 {code} 历史日线失败") from last_error


def fetch_histories(codes: list[str]) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=HISTORY_FETCH_WORKERS) as executor:
        future_map = {executor.submit(fetch_stock_history, code): code for code in codes}
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                history_code, history_df = future.result()
                histories[history_code] = history_df
            except Exception as exc:  # pragma: no cover - 依赖网络
                print(f"历史日线拉取失败 {code}: {exc}")
                histories[code] = pd.DataFrame()
    return histories


def get_quote_on_or_before(history_df: pd.DataFrame, trade_date: pd.Timestamp) -> tuple[pd.Series | None, bool]:
    if history_df.empty:
        return None, False
    if trade_date < history_df.index.min():
        return None, False
    effective_date = history_df.index.asof(trade_date)
    if pd.isna(effective_date):
        return None, False
    row = history_df.loc[effective_date]
    return row, pd.Timestamp(effective_date) == trade_date


def build_open_price_check(
    candidate_book: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    threshold: float = OPEN_PRICE_MISMATCH_THRESHOLD,
) -> pd.DataFrame:
    columns = [
        "日期",
        "股票代码",
        "股票简称",
        "基础代码",
        "CSV开盘价不复权",
        "日线开盘价不复权",
        "开盘价偏差",
        "开盘价偏差绝对值",
        "阈值",
        "说明",
    ]
    if candidate_book.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for _, record in candidate_book.iterrows():
        trade_date = pd.Timestamp(record["日期"]).normalize()
        code = str(record.get("基础代码", ""))
        csv_open = pd.to_numeric(record.get("开盘价:不复权今日"), errors="coerce")
        history_df = histories.get(code, pd.DataFrame())
        quote, exact_match = get_quote_on_or_before(history_df, trade_date)
        if quote is None or not exact_match:
            rows.append(
                {
                    "日期": trade_date.strftime("%Y%m%d"),
                    "股票代码": record.get("股票代码", ""),
                    "股票简称": record.get("股票简称", ""),
                    "基础代码": code,
                    "CSV开盘价不复权": csv_open,
                    "日线开盘价不复权": pd.NA,
                    "开盘价偏差": pd.NA,
                    "开盘价偏差绝对值": pd.NA,
                    "阈值": threshold,
                    "说明": "缺少当日日线，无法校验",
                }
            )
            continue

        daily_open = pd.to_numeric(quote.get("open"), errors="coerce")
        if pd.isna(csv_open) or pd.isna(daily_open) or float(daily_open) <= 0:
            rows.append(
                {
                    "日期": trade_date.strftime("%Y%m%d"),
                    "股票代码": record.get("股票代码", ""),
                    "股票简称": record.get("股票简称", ""),
                    "基础代码": code,
                    "CSV开盘价不复权": csv_open,
                    "日线开盘价不复权": daily_open,
                    "开盘价偏差": pd.NA,
                    "开盘价偏差绝对值": pd.NA,
                    "阈值": threshold,
                    "说明": "开盘价缺失或非法，无法校验",
                }
            )
            continue

        deviation = float(csv_open) / float(daily_open) - 1
        if abs(deviation) > threshold:
            rows.append(
                {
                    "日期": trade_date.strftime("%Y%m%d"),
                    "股票代码": record.get("股票代码", ""),
                    "股票简称": record.get("股票简称", ""),
                    "基础代码": code,
                    "CSV开盘价不复权": round(float(csv_open), 4),
                    "日线开盘价不复权": round(float(daily_open), 4),
                    "开盘价偏差": round(deviation, 6),
                    "开盘价偏差绝对值": round(abs(deviation), 6),
                    "阈值": threshold,
                    "说明": "CSV开盘价与日线不复权开盘价不一致",
                }
            )

    report_df = pd.DataFrame(rows, columns=columns)
    if not report_df.empty:
        report_df = report_df.sort_values(["日期", "开盘价偏差绝对值"], ascending=[True, False], kind="stable")
    return report_df.reset_index(drop=True)


def build_trading_calendar(candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    calendar_dates = set(candidate_book["日期"].tolist())
    for history_df in histories.values():
        if history_df.empty:
            continue
        calendar_dates.update(history_df.index.tolist())
    return sorted(pd.Timestamp(date).normalize() for date in calendar_dates)


def create_position(record: dict[str, Any], budget: float) -> Position:
    if budget <= 0:
        raise ValueError("开仓预算必须大于 0")
    raw_entry_price = float(record["开盘价:不复权今日"])
    exec_entry_price = raw_entry_price * (1 + SLIPPAGE_RATE)
    shares = budget / (exec_entry_price * (1 + BUY_COMMISSION_RATE))
    return Position(
        code=record["基础代码"],
        raw_code=str(record["股票代码"]),
        name=str(record["股票简称"]),
        entry_date=pd.Timestamp(record["日期"]),
        entry_price_raw=raw_entry_price,
        entry_price_exec=exec_entry_price,
        allocated_cash=budget,
        shares=shares,
        signal_rank=float(record[BUY_RANK_COLUMN]),
        last_close=float(record.get("收盘价:不复权今日", raw_entry_price)),
    )


def estimate_position_value(position: Position, quote: pd.Series | None) -> float:
    if quote is not None and not pd.isna(quote.get("close")):
        position.last_close = float(quote["close"])
    return position.shares * position.last_close


def run_backtest(candidate_book: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidate_book.empty:
        raise ValueError("候选池为空，无法回测")

    candidate_groups = {
        pd.Timestamp(trade_date): group.sort_values(BUY_RANK_COLUMN, kind="stable").to_dict("records")
        for trade_date, group in candidate_book.groupby("日期")
    }
    trading_calendar = build_trading_calendar(candidate_book, histories)
    trading_calendar = [trade_date for trade_date in trading_calendar if trade_date >= candidate_book["日期"].min()]

    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    trade_records: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []

    for trade_date in trading_calendar:
        open_slots = MAX_POSITIONS - len(positions)
        if open_slots > 0 and cash > 1e-8 and trade_date in candidate_groups:
            buyable_records: list[dict[str, Any]] = []
            for record in candidate_groups[trade_date]:
                if len(buyable_records) >= open_slots:
                    break
                history_df = histories.get(record["基础代码"], pd.DataFrame())
                quote, exact_match = get_quote_on_or_before(history_df, trade_date)
                if quote is None or not exact_match:
                    continue
                if pd.isna(record.get("开盘价:不复权今日")) or float(record["开盘价:不复权今日"]) <= 0:
                    continue
                buyable_records.append(record)

            if buyable_records:
                budget_per_trade = cash / len(buyable_records)
                if budget_per_trade <= 0:
                    buyable_records = []
                for record in buyable_records:
                    position = create_position(record, budget_per_trade)
                    cash -= budget_per_trade
                    positions.append(position)

        positions_to_close: list[tuple[Position, float]] = []
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, exact_match = get_quote_on_or_before(history_df, trade_date)
            if quote is not None and not pd.isna(quote.get("close")):
                position.last_close = float(quote["close"])
            if trade_date <= position.entry_date or quote is None or not exact_match:
                continue
            prev_close = float(quote.get("prev_close")) if not pd.isna(quote.get("prev_close")) else math.nan
            close_price = float(quote.get("close")) if not pd.isna(quote.get("close")) else math.nan
            if not is_limit_up_close(position.code, position.name, close_price, prev_close):
                positions_to_close.append((position, close_price))

        for position, close_price in positions_to_close:
            exec_exit_price = close_price * (1 - SLIPPAGE_RATE)
            net_proceeds = position.shares * exec_exit_price * (1 - SELL_COMMISSION_RATE - STAMP_DUTY_RATE)
            cash += net_proceeds

            trade_records.append(
                {
                    "股票代码": position.raw_code,
                    "基础代码": position.code,
                    "股票简称": position.name,
                    "买入日期": position.entry_date.strftime("%Y%m%d"),
                    "卖出日期": trade_date.strftime("%Y%m%d"),
                    "买入原价": round(position.entry_price_raw, 4),
                    "买入执行价": round(position.entry_price_exec, 4),
                    "卖出执行价": round(exec_exit_price, 4),
                    "信号热度排名": position.signal_rank,
                    "分配资金": round(position.allocated_cash, 2),
                    "卖出回款": round(net_proceeds, 2),
                    "单笔净收益率": round(net_proceeds / position.allocated_cash - 1, 6),
                    "持有自然日": (trade_date - position.entry_date).days,
                }
            )
            positions.remove(position)

        position_value = 0.0
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, _ = get_quote_on_or_before(history_df, trade_date)
            position_value += estimate_position_value(position, quote)

        equity = cash + position_value
        equity_records.append(
            {
                "日期": trade_date,
                "现金": round(cash, 2),
                "持仓市值": round(position_value, 2),
                "总权益": round(equity, 2),
                "持仓数量": len(positions),
            }
        )

    equity_df = pd.DataFrame(equity_records)
    equity_df["净值"] = equity_df["总权益"] / INITIAL_CAPITAL
    equity_df["历史高点"] = equity_df["净值"].cummax()
    equity_df["回撤"] = equity_df["净值"] / equity_df["历史高点"] - 1

    trade_df = pd.DataFrame(trade_records)
    summary = {
        "初始资金": INITIAL_CAPITAL,
        "期末权益": float(equity_df.iloc[-1]["总权益"]),
        "期末净值": float(equity_df.iloc[-1]["净值"]),
        "总收益率": float(equity_df.iloc[-1]["净值"] - 1),
        "最大回撤": float(equity_df["回撤"].min()),
        "交易笔数": int(len(trade_df)),
        "胜率": float((trade_df["单笔净收益率"] > 0).mean()) if not trade_df.empty else math.nan,
        "平均单笔净收益率": float(trade_df["单笔净收益率"].mean()) if not trade_df.empty else math.nan,
        "收益回撤比": float((equity_df.iloc[-1]["净值"] - 1) / abs(equity_df["回撤"].min()))
        if equity_df["回撤"].min() < 0
        else math.nan,
        "未平仓数量": len(positions),
    }
    summary_df = pd.DataFrame([summary])
    return trade_df, equity_df, summary_df


def build_scan_configs() -> list[StrategyConfig]:
    configs: list[StrategyConfig] = [
        StrategyConfig(name="基线"),
        StrategyConfig(
            name="统计脚本过滤",
            named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
        ),
        StrategyConfig(name="非ST", named_conditions=("not_st",)),
        StrategyConfig(name="连板<=0", buy_filters={"连续涨停天数昨日": {"max": 0}}),
    ]

    for threshold in [1.0, 1.2, 1.5, 2.0]:
        configs.append(StrategyConfig(name=f"量比>={threshold}", buy_filters={"量比": {"min": threshold}}))

    for threshold in [10, 15, 20, 30]:
        configs.append(
            StrategyConfig(name=f"换手率昨日>={threshold}", buy_filters={"换手率昨日": {"min": threshold}})
        )

    for threshold in [0, 200, 500, 1000]:
        configs.append(
            StrategyConfig(
                name=f"DDX>={threshold}",
                buy_filters={"大单动向(ddx值)昨日": {"min": threshold}},
            )
        )

    for minimum in [-6.0, -5.0, -4.0]:
        configs.append(
            StrategyConfig(
                name=f"竞价涨幅[{minimum},0]",
                buy_filters={"竞价涨幅今日": {"min": minimum, "max": 0.0}},
            )
        )

    for threshold in [0, 100, 300, 500]:
        configs.append(
            StrategyConfig(
                name=f"市场20高低差>={threshold}",
                market_filter={"min_high20_minus_low20": threshold},
            )
        )

    configs.extend(
        [
            StrategyConfig(
                name="统计过滤+非ST",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev", "not_st"),
            ),
            StrategyConfig(
                name="统计过滤+量比>=1.2",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                buy_filters={"量比": {"min": 1.2}},
            ),
            StrategyConfig(
                name="统计过滤+DDX>=0",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                buy_filters={"大单动向(ddx值)昨日": {"min": 0}},
            ),
            StrategyConfig(
                name="统计过滤+连板<=0",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                buy_filters={"连续涨停天数昨日": {"max": 0}},
            ),
            StrategyConfig(
                name="统计过滤+竞价涨幅[-6,0]",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                buy_filters={"竞价涨幅今日": {"min": -6.0, "max": 0.0}},
            ),
            StrategyConfig(
                name="统计过滤+市场20高低差>=0",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                market_filter={"min_high20_minus_low20": 0},
            ),
            StrategyConfig(
                name="统计过滤+市场20高低差>=300",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                market_filter={"min_high20_minus_low20": 300},
            ),
            StrategyConfig(
                name="统计过滤+量比>=1.2+市场>=0",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                buy_filters={"量比": {"min": 1.2}},
                market_filter={"min_high20_minus_low20": 0},
            ),
            StrategyConfig(
                name="统计过滤+DDX>=0+市场>=0",
                named_conditions=("yesterday_body_lt_prev", "yesterday_volume_gt_prev"),
                buy_filters={"大单动向(ddx值)昨日": {"min": 0}},
                market_filter={"min_high20_minus_low20": 0},
            ),
        ]
    )
    return configs


def run_strategy_scan(
    signal_df: pd.DataFrame,
    breadth_df: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    configs: list[StrategyConfig],
) -> pd.DataFrame:
    scan_rows: list[dict[str, Any]] = []
    for config in configs:
        candidate_book = build_candidate_book_with_config(
            signal_df=signal_df,
            breadth_df=breadth_df,
            buy_filters=config.buy_filters,
            named_conditions=config.named_conditions,
            market_filter=config.market_filter,
            sort_by=config.sort_by,
            max_positions=MAX_POSITIONS,
        )

        if candidate_book.empty:
            scan_rows.append(
                {
                    "策略名": config.name,
                    "命名条件": "; ".join(config.named_conditions),
                    "买入过滤": format_filter_rules(config.buy_filters),
                    "市场过滤": format_market_filter(config.market_filter),
                    "候选池行数": 0,
                    "候选交易日": 0,
                    "初始资金": INITIAL_CAPITAL,
                    "期末权益": math.nan,
                    "期末净值": math.nan,
                    "总收益率": math.nan,
                    "最大回撤": math.nan,
                    "交易笔数": 0,
                    "胜率": math.nan,
                    "平均单笔净收益率": math.nan,
                    "收益回撤比": math.nan,
                    "未平仓数量": 0,
                }
            )
            continue

        trade_df, equity_df, summary_df = run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        row.update(
            {
                "策略名": config.name,
                "命名条件": "; ".join(config.named_conditions),
                "买入过滤": format_filter_rules(config.buy_filters),
                "市场过滤": format_market_filter(config.market_filter),
                "候选池行数": int(len(candidate_book)),
                "候选交易日": int(candidate_book["日期"].nunique()),
            }
        )
        scan_rows.append(row)

    scan_df = pd.DataFrame(scan_rows)
    return scan_df.sort_values(["收益回撤比", "总收益率", "最大回撤"], ascending=[False, False, False])


def choose_best_config(scan_df: pd.DataFrame, configs: list[StrategyConfig]) -> StrategyConfig:
    valid_df = scan_df.dropna(subset=["收益回撤比", "总收益率", "最大回撤"]).copy()
    if valid_df.empty:
        return configs[0]

    valid_df = valid_df[valid_df["交易笔数"] >= 150].copy()
    if valid_df.empty:
        valid_df = scan_df.dropna(subset=["收益回撤比", "总收益率", "最大回撤"]).copy()

    best_name = valid_df.iloc[0]["策略名"]
    for config in configs:
        if config.name == best_name:
            return config
    return configs[0]


def append_strategy_metadata(summary_df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    result_df = summary_df.copy()
    result_df.insert(0, "策略名", config.name)
    result_df.insert(1, "命名条件", "; ".join(config.named_conditions))
    result_df.insert(2, "买入过滤", format_filter_rules(config.buy_filters))
    result_df.insert(3, "市场过滤", format_market_filter(config.market_filter))
    result_df.insert(4, "排序方式", str(config.sort_by) if config.sort_by else f"{BUY_RANK_COLUMN}升序")
    return result_df


def run_config_backtest(
    signal_df: pd.DataFrame,
    breadth_df: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_book = build_candidate_book_with_config(
        signal_df=signal_df,
        breadth_df=breadth_df,
        buy_filters=config.buy_filters,
        named_conditions=config.named_conditions,
        market_filter=config.market_filter,
        sort_by=config.sort_by,
        max_positions=MAX_POSITIONS,
    )
    trade_df, equity_df, summary_df = run_backtest(candidate_book, histories)
    summary_df = append_strategy_metadata(summary_df, config)
    return candidate_book, trade_df, equity_df, summary_df


def save_plot(equity_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(equity_df["日期"], equity_df["净值"], color="#d94841", linewidth=2, label="策略净值")
    axes[0].set_title("竞价爬升策略净值曲线")
    axes[0].set_ylabel("净值")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(equity_df["日期"], equity_df["回撤"], 0, color="#4c78a8", alpha=0.35)
    axes[1].set_title("回撤")
    axes[1].set_ylabel("回撤")
    axes[1].set_xlabel("日期")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_results(
    label: str,
    candidate_book: pd.DataFrame,
    breadth_df: pd.DataFrame,
    latest_activity_df: pd.DataFrame | None,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    include_market_files: bool = False,
) -> None:
    candidate_columns = [column for column in EXPORT_CANDIDATE_COLUMNS if column in candidate_book.columns]
    candidate_book[candidate_columns].to_csv(f"{OUTPUT_STEM}-{label}候选池.csv", index=False, encoding="utf-8-sig")
    if include_market_files:
        breadth_df.to_csv(f"{OUTPUT_STEM}-市场宽度.csv", index=False, encoding="utf-8-sig")
        if latest_activity_df is not None and not latest_activity_df.empty:
            latest_activity_df.to_csv(f"{OUTPUT_STEM}-最新市场情绪.csv", index=False, encoding="utf-8-sig")
    trade_df.to_csv(f"{OUTPUT_STEM}-{label}交易明细.csv", index=False, encoding="utf-8-sig")
    export_equity_df = equity_df.copy()
    export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
    export_equity_df.to_csv(f"{OUTPUT_STEM}-{label}净值.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(f"{OUTPUT_STEM}-{label}摘要.csv", index=False, encoding="utf-8-sig")
    save_plot(equity_df, Path(f"{OUTPUT_STEM}-{label}净值.png"))


def main() -> None:
    signal_df = load_signal_data(INPUT_CSV)
    scan_signal_df = restrict_signal_universe(signal_df, SCAN_RANK_UNIVERSE_LIMIT)
    train_signal_df, valid_signal_df, split_date = split_signal_data_by_ratio(scan_signal_df, TRAIN_RATIO)
    breadth_df = fetch_market_breadth_history(MARKET_BREADTH_SYMBOL)
    candidate_book = build_candidate_book(scan_signal_df, breadth_df)
    all_signal_codes = sorted(scan_signal_df["基础代码"].dropna().astype(str).unique().tolist())
    print(f"候选池行数: {len(candidate_book)}")
    print(f"扫描候选 universe: 每日热度前 {SCAN_RANK_UNIVERSE_LIMIT} 名")
    print(f"训练期截止前一日: {(split_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
    print(f"验证期起始日: {split_date.strftime('%Y-%m-%d')}")
    print(f"需要补日线的股票数: {len(all_signal_codes)}")

    histories = fetch_histories(all_signal_codes)
    latest_activity_df: pd.DataFrame | None
    try:
        latest_activity_df = fetch_latest_market_activity()
    except Exception as exc:  # pragma: no cover - 依赖网络
        print(f"最新市场情绪拉取失败: {exc}")
        latest_activity_df = None

    base_config = StrategyConfig(name="基线")
    candidate_book, trade_df, equity_df, summary_df = run_config_backtest(
        signal_df=scan_signal_df,
        breadth_df=breadth_df,
        histories=histories,
        config=base_config,
    )
    export_results(
        label="策略",
        candidate_book=candidate_book,
        breadth_df=breadth_df,
        latest_activity_df=latest_activity_df,
        trade_df=trade_df,
        equity_df=equity_df,
        summary_df=summary_df,
        include_market_files=True,
    )

    scan_configs = build_scan_configs()
    train_scan_df = run_strategy_scan(train_signal_df, breadth_df, histories, scan_configs)
    train_scan_df.to_csv(f"{OUTPUT_STEM}-训练期策略扫描.csv", index=False, encoding="utf-8-sig")

    best_config = choose_best_config(train_scan_df, scan_configs)
    best_candidate_book, best_trade_df, best_equity_df, best_summary_df = run_config_backtest(
        signal_df=scan_signal_df,
        breadth_df=breadth_df,
        histories=histories,
        config=best_config,
    )
    export_results(
        label="策略改良版",
        candidate_book=best_candidate_book,
        breadth_df=breadth_df,
        latest_activity_df=latest_activity_df,
        trade_df=best_trade_df,
        equity_df=best_equity_df,
        summary_df=best_summary_df,
    )

    valid_base_candidate_book, valid_base_trade_df, valid_base_equity_df, valid_base_summary_df = run_config_backtest(
        signal_df=valid_signal_df,
        breadth_df=breadth_df,
        histories=histories,
        config=base_config,
    )
    export_results(
        label="验证期基线",
        candidate_book=valid_base_candidate_book,
        breadth_df=breadth_df,
        latest_activity_df=latest_activity_df,
        trade_df=valid_base_trade_df,
        equity_df=valid_base_equity_df,
        summary_df=valid_base_summary_df,
    )

    valid_best_candidate_book, valid_best_trade_df, valid_best_equity_df, valid_best_summary_df = run_config_backtest(
        signal_df=valid_signal_df,
        breadth_df=breadth_df,
        histories=histories,
        config=best_config,
    )
    export_results(
        label="验证期改良版",
        candidate_book=valid_best_candidate_book,
        breadth_df=breadth_df,
        latest_activity_df=latest_activity_df,
        trade_df=valid_best_trade_df,
        equity_df=valid_best_equity_df,
        summary_df=valid_best_summary_df,
    )

    validation_compare_df = pd.concat(
        [
            valid_base_summary_df.assign(样本区间="验证期", 对比组="基线"),
            valid_best_summary_df.assign(样本区间="验证期", 对比组="改良版"),
        ],
        ignore_index=True,
    )
    validation_compare_df.to_csv(f"{OUTPUT_STEM}-验证期对比摘要.csv", index=False, encoding="utf-8-sig")

    if FACTOR_INPUT_CSV.exists():
        factor_signal_df = load_factor_signal_data(FACTOR_INPUT_CSV)
        if not factor_signal_df.empty:
            factor_codes = sorted(factor_signal_df["基础代码"].dropna().astype(str).unique().tolist())
            missing_codes = [code for code in factor_codes if code not in histories]
            if missing_codes:
                histories.update(fetch_histories(missing_codes))

            factor_candidate_book, factor_trade_df, factor_equity_df, factor_summary_df = run_config_backtest(
                signal_df=factor_signal_df,
                breadth_df=breadth_df,
                histories=histories,
                config=NEW_FACTOR_CONFIG,
            )
            export_results(
                label="新因子策略版",
                candidate_book=factor_candidate_book,
                breadth_df=breadth_df,
                latest_activity_df=latest_activity_df,
                trade_df=factor_trade_df,
                equity_df=factor_equity_df,
                summary_df=factor_summary_df,
            )
            print("\n新因子策略版:")
            print(factor_summary_df.to_string(index=False))

            ratio_candidate_book, ratio_trade_df, ratio_equity_df, ratio_summary_df = run_config_backtest(
                signal_df=factor_signal_df,
                breadth_df=breadth_df,
                histories=histories,
                config=AUCTION_RATIO_RANK_CONFIG,
            )
            export_results(
                label="竞昨比排序版",
                candidate_book=ratio_candidate_book,
                breadth_df=breadth_df,
                latest_activity_df=latest_activity_df,
                trade_df=ratio_trade_df,
                equity_df=ratio_equity_df,
                summary_df=ratio_summary_df,
            )
            print("\n竞昨比排序版:")
            print(ratio_summary_df.to_string(index=False))
        else:
            print("\n扩展因子文件存在，但没有可用于新因子策略版的样本")
    else:
        print(f"\n未找到扩展因子文件: {FACTOR_INPUT_CSV}")

    print("策略回测完成")
    print(summary_df.to_string(index=False))
    print("\n参数扫描前五名:")
    print(train_scan_df.head(5).to_string(index=False))
    print("\n当前选中的改良版策略:")
    print(best_summary_df.to_string(index=False))
    print("\n验证期对比:")
    print(validation_compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
