from __future__ import annotations

import importlib.util
import io
import math
import sys
import time
import urllib3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
CACHE_DIR = BASE_DIR / "cache" / "sw_first"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class SectorExperimentConfig:
    name: str
    industry_return_min: float | None = None
    industry_return_rank_max: int | None = None
    industry_amount_rank_max: int | None = None
    peer_count_min: int | None = None
    sort_by: tuple[tuple[str, bool], ...] | None = None


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("auction_backtest", BACKTEST_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载回测模块: {BACKTEST_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    max_retries: int = 3,
    timeout: int = 30,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=HEADERS,
                verify=False,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(attempt)
    raise RuntimeError(f"请求失败: {url}") from last_error


def read_cached_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_trade_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return parsed.dt.normalize()


def fetch_first_level_index_list(session: requests.Session) -> pd.DataFrame:
    cache_path = CACHE_DIR / "sw_first_index_list.csv"
    cached = read_cached_csv(cache_path)
    if not cached.empty:
        return cached

    response = request_with_retry(
        session,
        "https://www.swsresearch.com/institute-sw/api/index_publish/current/",
        params={
            "page": "1",
            "page_size": "100",
            "indextype": "一级行业",
        },
    )
    data_json = response.json()
    result_df = pd.DataFrame(data_json["data"]["results"])[["swindexcode", "swindexname"]].copy()
    result_df = result_df.drop_duplicates().reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return result_df


def fetch_first_level_components(session: requests.Session, first_index_df: pd.DataFrame) -> pd.DataFrame:
    cache_path = CACHE_DIR / "sw_first_components.csv"
    cached = read_cached_csv(cache_path)
    if not cached.empty:
        cached["stockcode"] = cached["stockcode"].astype(str).str.zfill(6)
        return cached

    frames: list[pd.DataFrame] = []
    for _, row in first_index_df.iterrows():
        response = request_with_retry(
            session,
            "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/",
            params={
                "swindexcode": row["swindexcode"],
                "page": "1",
                "page_size": "10000",
            },
        )
        component_df = pd.DataFrame(response.json()["data"]["results"])
        if component_df.empty:
            continue
        component_df = component_df[["stockcode", "stockname"]].copy()
        component_df["stockcode"] = component_df["stockcode"].astype(str).str.zfill(6)
        component_df["first_index_code"] = row["swindexcode"]
        component_df["first_name"] = row["swindexname"]
        frames.append(component_df)

    if not frames:
        raise RuntimeError("未获取到申万一级行业成分股数据")

    result_df = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return result_df


def fetch_stock_industry_history(session: requests.Session) -> pd.DataFrame:
    cache_path = CACHE_DIR / "sw_stock_industry_history.csv"
    cached = read_cached_csv(cache_path)
    if not cached.empty:
        cached["symbol"] = cached["symbol"].astype(str).str.zfill(6)
        cached["start_date"] = pd.to_datetime(cached["start_date"])
        cached["update_time"] = pd.to_datetime(cached["update_time"])
        return cached

    response = request_with_retry(
        session,
        "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls",
        timeout=60,
    )
    history_df = pd.read_excel(
        io.BytesIO(response.content),
        dtype={"股票代码": "str", "行业代码": "str"},
    )
    history_df = history_df.rename(
        columns={
            "股票代码": "symbol",
            "计入日期": "start_date",
            "行业代码": "industry_code",
            "更新日期": "update_time",
        }
    )
    history_df["symbol"] = history_df["symbol"].astype(str).str.zfill(6)
    history_df["start_date"] = pd.to_datetime(history_df["start_date"])
    history_df["update_time"] = pd.to_datetime(history_df["update_time"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return history_df


def build_histcode_to_first_map(
    industry_history_df: pd.DataFrame,
    first_component_df: pd.DataFrame,
) -> pd.DataFrame:
    latest_df = (
        industry_history_df.sort_values(["symbol", "start_date"])
        .groupby("symbol", as_index=False)
        .tail(1)
    )
    merged = latest_df.merge(
        first_component_df[["stockcode", "first_index_code", "first_name"]],
        left_on="symbol",
        right_on="stockcode",
        how="inner",
    )
    ambiguity_df = (
        merged.groupby("industry_code")[["first_index_code", "first_name"]]
        .nunique()
        .reset_index()
    )
    ambiguous = ambiguity_df[
        (ambiguity_df["first_index_code"] > 1) | (ambiguity_df["first_name"] > 1)
    ]
    if not ambiguous.empty:
        raise RuntimeError("历史行业代码到一级行业映射存在歧义")

    mapping_df = (
        merged.groupby("industry_code", as_index=False)[["first_index_code", "first_name"]]
        .first()
        .rename(
            columns={
                "first_index_code": "申万一级行业代码",
                "first_name": "申万一级行业",
            }
        )
    )
    return mapping_df


def annotate_first_industry(
    signal_df: pd.DataFrame,
    industry_history_df: pd.DataFrame,
    histcode_map_df: pd.DataFrame,
) -> pd.DataFrame:
    left_df = signal_df[["基础代码", "日期"]].copy()
    left_df = left_df.rename(columns={"基础代码": "symbol", "日期": "trade_date"})
    left_df["row_id"] = signal_df.index
    left_df = left_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    right_df = industry_history_df[["symbol", "start_date", "industry_code"]].copy()
    right_df = right_df.sort_values(["start_date", "symbol"]).reset_index(drop=True)

    merged_df = pd.merge_asof(
        left_df,
        right_df,
        left_on="trade_date",
        right_on="start_date",
        by="symbol",
        direction="backward",
    )

    annotated_df = (
        signal_df.reset_index()
        .merge(merged_df[["row_id", "industry_code"]], left_on="index", right_on="row_id", how="left")
        .drop(columns=["index", "row_id"])
    )
    annotated_df = annotated_df.merge(histcode_map_df, on="industry_code", how="left")
    return annotated_df


def fetch_first_level_daily_analysis(
    session: requests.Session,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"sw_first_daily_{start_date}_{end_date}.csv"
    cached = read_cached_csv(cache_path)
    if not cached.empty:
        cached["日期"] = normalize_trade_date(cached["日期"])
        numeric_columns = [
            "申万一级行业涨跌幅",
            "申万一级行业换手率",
            "申万一级行业成交额占比",
            "申万一级行业涨跌幅排名",
            "申万一级行业成交额占比排名",
            "申万一级行业换手率排名",
        ]
        for column in numeric_columns:
            cached[column] = pd.to_numeric(cached[column], errors="coerce")
        return cached

    url = "https://www.swsresearch.com/institute-sw/api/index_analysis/index_analysis_report/"
    page_size = 50
    params = {
        "page": "1",
        "page_size": str(page_size),
        "index_type": "一级行业",
        "start_date": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
        "end_date": f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
        "type": "DAY",
        "swindexcode": "all",
    }

    first_response = request_with_retry(session, url, params=params)
    first_json = first_response.json()
    total_count = int(first_json["data"]["count"])
    total_pages = math.ceil(total_count / page_size)
    results = list(first_json["data"]["results"])

    for page in range(2, total_pages + 1):
        params["page"] = str(page)
        response = request_with_retry(session, url, params=params)
        results.extend(response.json()["data"]["results"])

    daily_df = pd.DataFrame(results)
    if daily_df.empty:
        raise RuntimeError("未获取到申万一级行业日线数据")

    daily_df = daily_df.rename(
        columns={
            "swindexcode": "申万一级行业代码",
            "swindexname": "申万一级行业",
            "bargaindate": "日期",
            "markup": "申万一级行业涨跌幅",
            "turnoverrate": "申万一级行业换手率",
            "bargainsumrate": "申万一级行业成交额占比",
        }
    )
    keep_columns = [
        "申万一级行业代码",
        "申万一级行业",
        "日期",
        "申万一级行业涨跌幅",
        "申万一级行业换手率",
        "申万一级行业成交额占比",
    ]
    daily_df = daily_df[keep_columns].copy()
    daily_df["日期"] = normalize_trade_date(daily_df["日期"])
    for column in ["申万一级行业涨跌幅", "申万一级行业换手率", "申万一级行业成交额占比"]:
        daily_df[column] = pd.to_numeric(daily_df[column], errors="coerce")

    daily_df["申万一级行业涨跌幅排名"] = (
        daily_df.groupby("日期")["申万一级行业涨跌幅"].rank(ascending=False, method="min")
    )
    daily_df["申万一级行业成交额占比排名"] = (
        daily_df.groupby("日期")["申万一级行业成交额占比"].rank(ascending=False, method="min")
    )
    daily_df["申万一级行业换手率排名"] = (
        daily_df.groupby("日期")["申万一级行业换手率"].rank(ascending=False, method="min")
    )
    daily_df = daily_df.sort_values(["日期", "申万一级行业涨跌幅排名", "申万一级行业代码"]).reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return daily_df


def load_breadth_data(strategy_mod) -> pd.DataFrame:
    local_path = Path(f"{strategy_mod.OUTPUT_STEM}-市场宽度.csv")
    if local_path.exists():
        breadth_df = strategy_mod.read_csv_with_fallback(local_path)
        breadth_df["日期"] = pd.to_datetime(breadth_df["日期"]).dt.normalize()
        return breadth_df
    return strategy_mod.fetch_market_breadth_history()


def build_base_candidate_universe(strategy_mod, signal_df: pd.DataFrame, breadth_df: pd.DataFrame) -> pd.DataFrame:
    config = strategy_mod.AUCTION_RATIO_RANK_CONFIG
    candidates = signal_df.copy()
    if not breadth_df.empty:
        candidates = candidates.merge(breadth_df, on="日期", how="left")

    if config.named_conditions:
        candidates = strategy_mod.apply_named_conditions(candidates, config.named_conditions)
    if config.buy_filters:
        candidates = strategy_mod.apply_filter_rules(candidates, config.buy_filters)
    if config.market_filter:
        threshold_20 = config.market_filter.get("min_high20_minus_low20")
        if threshold_20 is not None and "市场20日高低差" in candidates.columns:
            candidates = candidates[candidates["市场20日高低差"] >= threshold_20].copy()
        threshold_60 = config.market_filter.get("min_high60_minus_low60")
        if threshold_60 is not None and "市场60日高低差" in candidates.columns:
            candidates = candidates[candidates["市场60日高低差"] >= threshold_60].copy()

    candidates = candidates.copy()
    candidates["一级行业候选数"] = 0
    valid_mask = candidates["申万一级行业代码"].notna()
    if valid_mask.any():
        counted = (
            candidates.loc[valid_mask]
            .groupby(["日期", "申万一级行业代码"])["基础代码"]
            .transform("size")
        )
        candidates.loc[valid_mask, "一级行业候选数"] = counted.astype(int)
    return candidates.reset_index(drop=True)


def build_experiment_configs(strategy_mod) -> list[SectorExperimentConfig]:
    base_sort = strategy_mod.AUCTION_RATIO_RANK_CONFIG.sort_by or ()
    heat_column = strategy_mod.BUY_RANK_COLUMN
    configs: list[SectorExperimentConfig] = [
        SectorExperimentConfig(name="基线", sort_by=base_sort),
        SectorExperimentConfig(name="一级行业涨幅>0", industry_return_min=0.0, sort_by=base_sort),
    ]
    for rank in range(1, 16):
        configs.append(
            SectorExperimentConfig(
                name=f"一级行业涨幅排名<={rank}",
                industry_return_rank_max=rank,
                sort_by=base_sort,
            )
        )
    configs.extend(
        [
            SectorExperimentConfig(
                name="一级行业涨幅排名<=5+涨幅>0",
                industry_return_min=0.0,
                industry_return_rank_max=5,
                sort_by=base_sort,
            ),
            SectorExperimentConfig(
                name="一级行业涨幅排名<=6+涨幅>0",
                industry_return_min=0.0,
                industry_return_rank_max=6,
                sort_by=base_sort,
            ),
            SectorExperimentConfig(name="一级行业成交额占比排名<=10", industry_amount_rank_max=10, sort_by=base_sort),
            SectorExperimentConfig(name="一级行业成交额占比排名<=5", industry_amount_rank_max=5, sort_by=base_sort),
            SectorExperimentConfig(name="一级行业候选数>=2", peer_count_min=2, sort_by=base_sort),
            SectorExperimentConfig(
                name="一级行业候选数>=2+涨幅>0",
                peer_count_min=2,
                industry_return_min=0.0,
                sort_by=base_sort,
            ),
            SectorExperimentConfig(
                name="一级行业候选数>=2+涨幅排名<=10",
                peer_count_min=2,
                industry_return_rank_max=10,
                sort_by=base_sort,
            ),
            SectorExperimentConfig(
                name="行业涨幅优先排序",
                sort_by=(
                    ("申万一级行业涨跌幅排名", True),
                    ("竞昨成交比估算", False),
                    (heat_column, True),
                    ("基础代码", True),
                ),
            ),
            SectorExperimentConfig(
                name="行业成交额占比优先排序",
                sort_by=(
                    ("申万一级行业成交额占比排名", True),
                    ("竞昨成交比估算", False),
                    (heat_column, True),
                    ("基础代码", True),
                ),
            ),
            SectorExperimentConfig(
                name="行业候选数优先排序",
                sort_by=(
                    ("一级行业候选数", False),
                    ("竞昨成交比估算", False),
                    (heat_column, True),
                    ("基础代码", True),
                ),
            ),
            SectorExperimentConfig(
                name="行业候选数+行业涨幅排序",
                sort_by=(
                    ("一级行业候选数", False),
                    ("申万一级行业涨跌幅排名", True),
                    ("竞昨成交比估算", False),
                    (heat_column, True),
                    ("基础代码", True),
                ),
            ),
        ]
    )
    return configs


def build_candidate_book_for_config(
    strategy_mod,
    base_candidate_df: pd.DataFrame,
    config: SectorExperimentConfig,
) -> pd.DataFrame:
    candidates = base_candidate_df.copy()

    if config.industry_return_min is not None:
        candidates = candidates[
            pd.to_numeric(candidates["申万一级行业涨跌幅"], errors="coerce") >= config.industry_return_min
        ].copy()
    if config.industry_return_rank_max is not None:
        candidates = candidates[
            pd.to_numeric(candidates["申万一级行业涨跌幅排名"], errors="coerce") <= config.industry_return_rank_max
        ].copy()
    if config.industry_amount_rank_max is not None:
        candidates = candidates[
            pd.to_numeric(candidates["申万一级行业成交额占比排名"], errors="coerce") <= config.industry_amount_rank_max
        ].copy()
    if config.peer_count_min is not None:
        candidates = candidates[
            pd.to_numeric(candidates["一级行业候选数"], errors="coerce") >= config.peer_count_min
        ].copy()

    if candidates.empty:
        return candidates

    sort_columns = ["日期"]
    ascending = [True]
    for column, is_ascending in config.sort_by or ():
        if column not in candidates.columns:
            raise KeyError(f"排序列不存在: {column}")
        sort_columns.append(column)
        ascending.append(is_ascending)
    if "基础代码" not in sort_columns:
        sort_columns.append("基础代码")
        ascending.append(True)

    candidates = candidates.sort_values(sort_columns, ascending=ascending, kind="stable")
    candidate_book = candidates.groupby("日期", group_keys=False).head(strategy_mod.MAX_POSITIONS).copy()
    candidate_book.reset_index(drop=True, inplace=True)
    return candidate_book


def export_best_bundle(strategy_mod, label: str, candidate_book: pd.DataFrame, trade_df: pd.DataFrame, equity_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业联动实验"
    candidate_columns = [
        "日期",
        "股票代码",
        "股票简称",
        "基础代码",
        "申万一级行业代码",
        "申万一级行业",
        "申万一级行业涨跌幅",
        "申万一级行业涨跌幅排名",
        "申万一级行业成交额占比",
        "申万一级行业成交额占比排名",
        "一级行业候选数",
        "竞昨成交比估算",
        "个股热度排名昨日",
        "竞价匹配金额_openapi",
        "市场20日高低差",
        "市场60日高低差",
    ]
    candidate_columns = [column for column in candidate_columns if column in candidate_book.columns]
    export_candidate_df = candidate_book.copy()
    export_candidate_df["日期"] = pd.to_datetime(export_candidate_df["日期"]).dt.strftime("%Y%m%d")
    export_candidate_df[candidate_columns].to_csv(
        f"{output_prefix}-最佳策略候选池.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
    export_equity_df = equity_df.copy()
    export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
    export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")
    strategy_mod.save_plot(equity_df, Path(f"{output_prefix}-最佳策略净值.png"))
    pd.DataFrame([{"最佳策略": label}]).to_csv(
        f"{output_prefix}-最佳策略名称.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    strategy_mod = load_strategy_module()
    session = create_session()
    signal_df = strategy_mod.load_factor_signal_data(strategy_mod.FACTOR_INPUT_CSV)
    breadth_df = load_breadth_data(strategy_mod)

    first_index_df = fetch_first_level_index_list(session)
    first_component_df = fetch_first_level_components(session, first_index_df)
    industry_history_df = fetch_stock_industry_history(session)
    histcode_map_df = build_histcode_to_first_map(industry_history_df, first_component_df)

    signal_df = annotate_first_industry(signal_df, industry_history_df, histcode_map_df)
    start_date = signal_df["日期"].min().strftime("%Y%m%d")
    end_date = signal_df["日期"].max().strftime("%Y%m%d")
    first_daily_df = fetch_first_level_daily_analysis(session, start_date, end_date)
    signal_df = signal_df.merge(
        first_daily_df,
        on=["日期", "申万一级行业代码", "申万一级行业"],
        how="left",
    )

    coverage_rows = [
        {
            "阶段": "验证期信号",
            "行数": int(len(signal_df)),
            "一级行业映射覆盖率": round(float(signal_df["申万一级行业"].notna().mean()), 6),
            "一级行业日线覆盖率": round(float(signal_df["申万一级行业涨跌幅"].notna().mean()), 6),
        }
    ]

    base_candidate_df = build_base_candidate_universe(strategy_mod, signal_df, breadth_df)
    coverage_rows.append(
        {
            "阶段": "基线过滤后候选宇宙",
            "行数": int(len(base_candidate_df)),
            "一级行业映射覆盖率": round(float(base_candidate_df["申万一级行业"].notna().mean()), 6),
            "一级行业日线覆盖率": round(float(base_candidate_df["申万一级行业涨跌幅"].notna().mean()), 6),
        }
    )

    histories = strategy_mod.fetch_histories(
        sorted(base_candidate_df["基础代码"].dropna().astype(str).unique().tolist())
    )
    configs = build_experiment_configs(strategy_mod)

    rows: list[dict[str, object]] = []
    best_by_return: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    best_return = float("-inf")
    best_ratio = float("-inf")

    for config in configs:
        candidate_book = build_candidate_book_for_config(strategy_mod, base_candidate_df, config)
        if candidate_book.empty:
            rows.append(
                {
                    "策略名称": config.name,
                    "行业过滤": (
                        f"涨幅>={config.industry_return_min}; "
                        f"涨幅排名<={config.industry_return_rank_max}; "
                        f"成交额占比排名<={config.industry_amount_rank_max}; "
                        f"一级行业候选数>={config.peer_count_min}"
                    ),
                    "期末净值": float("nan"),
                    "总收益率": float("nan"),
                    "最大回撤": float("nan"),
                    "交易笔数": 0,
                    "胜率": float("nan"),
                    "收益回撤比": float("nan"),
                    "候选池行数": 0,
                    "候选交易日": 0,
                }
            )
            continue

        trade_df, equity_df, summary_df = strategy_mod.run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        row["策略名称"] = config.name
        row["行业过滤"] = (
            f"涨幅>={config.industry_return_min}; "
            f"涨幅排名<={config.industry_return_rank_max}; "
            f"成交额占比排名<={config.industry_amount_rank_max}; "
            f"一级行业候选数>={config.peer_count_min}"
        )
        row["候选池行数"] = int(len(candidate_book))
        row["候选交易日"] = int(candidate_book["日期"].nunique())
        rows.append(row)

        total_return = float(row["总收益率"])
        return_drawdown_ratio = float(row["收益回撤比"])
        if (total_return > best_return) or (
            math.isclose(total_return, best_return, rel_tol=1e-12, abs_tol=1e-12)
            and return_drawdown_ratio > best_ratio
        ):
            best_return = total_return
            best_ratio = return_drawdown_ratio
            best_by_return = (config.name, candidate_book, trade_df, equity_df, summary_df)

    result_df = pd.DataFrame(rows).sort_values(
        ["总收益率", "收益回撤比"],
        ascending=[False, False],
    ).reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage_rows)

    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业联动实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(f"{output_prefix}-覆盖率.csv", index=False, encoding="utf-8-sig")

    if best_by_return is not None:
        best_label, candidate_book, trade_df, equity_df, summary_df = best_by_return
        export_best_bundle(strategy_mod, best_label, candidate_book, trade_df, equity_df, summary_df)

    print("行业联动实验完成")
    print("覆盖率：")
    print(coverage_df.to_string(index=False))
    print("\n结果排名（按总收益率）：")
    print(
        result_df[
            [
                "策略名称",
                "期末净值",
                "总收益率",
                "最大回撤",
                "交易笔数",
                "胜率",
                "收益回撤比",
                "候选交易日",
            ]
        ].to_string(index=False)
    )
    if best_by_return is not None:
        print(f"\n最佳总收益策略: {best_by_return[0]}")


if __name__ == "__main__":
    main()
