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
SECTOR_EXPLORER_PATH = BASE_DIR / "竞价行业联动探索.py"
CACHE_DIR = BASE_DIR / "cache" / "sw_first"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class OpenSectorConfig:
    name: str
    open_rank_max: int | None = None
    open_change_min: float | None = None
    sort_by: tuple[tuple[str, bool], ...] | None = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_strategy_module():
    return load_module("auction_backtest_open_sector", BACKTEST_MODULE_PATH)


def load_sector_module():
    return load_module("sector_explorer_open_sector", SECTOR_EXPLORER_PATH)


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


def load_breadth_data(strategy_mod) -> pd.DataFrame:
    local_path = Path(f"{strategy_mod.OUTPUT_STEM}-市场宽度.csv")
    if local_path.exists():
        breadth_df = strategy_mod.read_csv_with_fallback(local_path)
        breadth_df["日期"] = pd.to_datetime(breadth_df["日期"]).dt.normalize()
        return breadth_df
    return strategy_mod.fetch_market_breadth_history()


def fetch_first_level_open_history(session: requests.Session, first_index_df: pd.DataFrame) -> pd.DataFrame:
    cache_path = CACHE_DIR / "sw_first_open_history.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        cached["日期"] = pd.to_datetime(cached["日期"]).dt.normalize()
        for column in ["收盘", "开盘", "最高", "最低", "成交量", "成交额", "申万一级行业开盘涨幅", "申万一级行业开盘涨幅排名"]:
            if column in cached.columns:
                cached[column] = pd.to_numeric(cached[column], errors="coerce")
        return cached

    frames: list[pd.DataFrame] = []
    for _, row in first_index_df.iterrows():
        response = request_with_retry(
            session,
            "https://www.swsresearch.com/institute-sw/api/index_publish/trend/",
            params={
                "swindexcode": row["swindexcode"],
                "period": "DAY",
            },
            timeout=60,
        )
        temp_df = pd.DataFrame(response.json()["data"])
        if temp_df.empty:
            continue
        temp_df = temp_df.rename(
            columns={
                "swindexcode": "申万一级行业代码",
                "bargaindate": "日期",
                "openindex": "开盘",
                "maxindex": "最高",
                "minindex": "最低",
                "closeindex": "收盘",
                "bargainamount": "成交量",
                "bargainsum": "成交额",
            }
        )
        temp_df["申万一级行业"] = row["swindexname"]
        keep_columns = [
            "申万一级行业代码",
            "申万一级行业",
            "日期",
            "收盘",
            "开盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
        ]
        temp_df = temp_df[keep_columns].copy()
        temp_df["日期"] = pd.to_datetime(temp_df["日期"]).dt.normalize()
        for column in ["收盘", "开盘", "最高", "最低", "成交量", "成交额"]:
            temp_df[column] = pd.to_numeric(temp_df[column], errors="coerce")
        frames.append(temp_df)

    if not frames:
        raise RuntimeError("未获取到申万一级行业开盘历史")

    history_df = pd.concat(frames, ignore_index=True)
    history_df = history_df.sort_values(["申万一级行业代码", "日期"]).reset_index(drop=True)
    history_df["前收"] = history_df.groupby("申万一级行业代码")["收盘"].shift(1)
    history_df["申万一级行业开盘涨幅"] = history_df["开盘"] / history_df["前收"] - 1
    history_df["申万一级行业开盘涨幅排名"] = history_df.groupby("日期")["申万一级行业开盘涨幅"].rank(
        ascending=False,
        method="min",
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return history_df


def prepare_signal_df(strategy_mod, sector_mod) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = strategy_mod.load_factor_signal_data(strategy_mod.FACTOR_INPUT_CSV)
    breadth_df = load_breadth_data(strategy_mod)

    session = sector_mod.create_session()
    first_index_df = sector_mod.fetch_first_level_index_list(session)
    first_component_df = sector_mod.fetch_first_level_components(session, first_index_df)
    industry_history_df = sector_mod.fetch_stock_industry_history(session)
    histcode_map_df = sector_mod.build_histcode_to_first_map(industry_history_df, first_component_df)
    signal_df = sector_mod.annotate_first_industry(signal_df, industry_history_df, histcode_map_df)
    if "申万一级行业代码" in signal_df.columns:
        signal_df["申万一级行业代码"] = signal_df["申万一级行业代码"].astype(str)

    open_history_df = fetch_first_level_open_history(create_session(), first_index_df)
    if "申万一级行业代码" in open_history_df.columns:
        open_history_df["申万一级行业代码"] = open_history_df["申万一级行业代码"].astype(str)
    keep_columns = [
        "申万一级行业代码",
        "申万一级行业",
        "日期",
        "申万一级行业开盘涨幅",
        "申万一级行业开盘涨幅排名",
    ]
    signal_df = signal_df.merge(
        open_history_df[keep_columns],
        on=["日期", "申万一级行业代码", "申万一级行业"],
        how="left",
    )
    return signal_df, breadth_df


def build_base_candidate_df(strategy_mod, sector_mod, signal_df: pd.DataFrame, breadth_df: pd.DataFrame) -> pd.DataFrame:
    return sector_mod.build_base_candidate_universe(strategy_mod, signal_df, breadth_df)


def build_configs(strategy_mod) -> list[OpenSectorConfig]:
    base_sort = strategy_mod.AUCTION_RATIO_RANK_CONFIG.sort_by or ()
    configs: list[OpenSectorConfig] = [
        OpenSectorConfig(name="开盘行业基线", sort_by=base_sort),
        OpenSectorConfig(name="行业开盘涨幅>0", open_change_min=0.0, sort_by=base_sort),
    ]
    for rank in range(1, 16):
        configs.append(
            OpenSectorConfig(
                name=f"行业开盘涨幅排名<={rank}",
                open_rank_max=rank,
                sort_by=base_sort,
            )
        )
    return configs


def build_candidate_book(strategy_mod, base_candidate_df: pd.DataFrame, config: OpenSectorConfig) -> pd.DataFrame:
    candidates = base_candidate_df.copy()
    if config.open_change_min is not None:
        candidates = candidates[
            pd.to_numeric(candidates["申万一级行业开盘涨幅"], errors="coerce") >= config.open_change_min
        ].copy()
    if config.open_rank_max is not None:
        candidates = candidates[
            pd.to_numeric(candidates["申万一级行业开盘涨幅排名"], errors="coerce") <= config.open_rank_max
        ].copy()
    if candidates.empty:
        return candidates

    sort_columns = ["日期"]
    ascending = [True]
    for column, is_ascending in config.sort_by or ():
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
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘联动实验"
    keep_columns = [
        "日期",
        "股票代码",
        "股票简称",
        "申万一级行业代码",
        "申万一级行业",
        "申万一级行业开盘涨幅",
        "申万一级行业开盘涨幅排名",
        "竞昨成交比估算",
        "个股热度排名昨日",
        "竞价匹配金额_openapi",
    ]
    keep_columns = [column for column in keep_columns if column in candidate_book.columns]
    export_candidate_df = candidate_book.copy()
    export_candidate_df["日期"] = pd.to_datetime(export_candidate_df["日期"]).dt.strftime("%Y%m%d")
    export_candidate_df[keep_columns].to_csv(
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
    sector_mod = load_sector_module()
    signal_df, breadth_df = prepare_signal_df(strategy_mod, sector_mod)
    base_candidate_df = build_base_candidate_df(strategy_mod, sector_mod, signal_df, breadth_df)
    histories = strategy_mod.fetch_histories(
        sorted(base_candidate_df["基础代码"].dropna().astype(str).unique().tolist())
    )

    rows: list[dict[str, object]] = []
    best_bundle: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    best_return = float("-inf")
    best_ratio = float("-inf")

    for config in build_configs(strategy_mod):
        candidate_book = build_candidate_book(strategy_mod, base_candidate_df, config)
        if candidate_book.empty:
            rows.append(
                {
                    "策略名称": config.name,
                    "开盘行业过滤": f"开盘行业涨幅>={config.open_change_min}; 开盘行业涨幅排名<={config.open_rank_max}",
                    "期末净值": float("nan"),
                    "总收益率": float("nan"),
                    "最大回撤": float("nan"),
                    "交易笔数": 0,
                    "胜率": float("nan"),
                    "收益回撤比": float("nan"),
                    "候选交易日": 0,
                }
            )
            continue

        trade_df, equity_df, summary_df = strategy_mod.run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        row["策略名称"] = config.name
        row["开盘行业过滤"] = f"开盘行业涨幅>={config.open_change_min}; 开盘行业涨幅排名<={config.open_rank_max}"
        row["候选交易日"] = int(candidate_book["日期"].nunique())
        rows.append(row)

        total_return = float(row["总收益率"])
        ratio = float(row["收益回撤比"])
        if (total_return > best_return) or (
            math.isclose(total_return, best_return, rel_tol=1e-12, abs_tol=1e-12) and ratio > best_ratio
        ):
            best_return = total_return
            best_ratio = ratio
            best_bundle = (config.name, candidate_book, trade_df, equity_df, summary_df)

    result_df = pd.DataFrame(rows).sort_values(["总收益率", "收益回撤比"], ascending=[False, False]).reset_index(drop=True)
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘联动实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        export_best_bundle(strategy_mod, *best_bundle)

    print("行业开盘联动实验完成")
    print(
        result_df[
            ["策略名称", "期末净值", "总收益率", "最大回撤", "交易笔数", "胜率", "收益回撤比", "候选交易日"]
        ].to_string(index=False)
    )
    if best_bundle is not None:
        print(f"\n最佳开盘行业策略: {best_bundle[0]}")


if __name__ == "__main__":
    main()
