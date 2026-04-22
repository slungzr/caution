from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"


@dataclass(frozen=True)
class MarketStateConfig:
    name: str
    market20_min: float | None = None
    market20_max: float | None = None
    market60_min: float | None = None
    market60_max: float | None = None


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("auction_backtest", BACKTEST_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载回测模块: {BACKTEST_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_breadth_data(strategy_mod) -> pd.DataFrame:
    local_path = Path(f"{strategy_mod.OUTPUT_STEM}-市场宽度.csv")
    if local_path.exists():
        breadth_df = strategy_mod.read_csv_with_fallback(local_path)
        breadth_df["日期"] = pd.to_datetime(breadth_df["日期"]).dt.normalize()
        return breadth_df
    return strategy_mod.fetch_market_breadth_history()


def build_market_state_configs():
    configs = [MarketStateConfig(name="基线", market20_min=0)]

    for threshold in [100, 200, 300, 400, 500, 600]:
        configs.append(MarketStateConfig(name=f"市场20高低差>={threshold}", market20_min=threshold))

    for threshold in [100, 200, 300, 400, 500]:
        configs.append(MarketStateConfig(name=f"市场60高低差>={threshold}", market20_min=0, market60_min=threshold))

    for upper in [199, 299, 399]:
        configs.append(MarketStateConfig(name=f"市场60高低差<={upper}", market20_min=0, market60_max=upper))

    configs.extend(
        [
            MarketStateConfig(name="市场60介于100_199", market20_min=0, market60_min=100, market60_max=199),
            MarketStateConfig(name="市场60介于100_299", market20_min=0, market60_min=100, market60_max=299),
            MarketStateConfig(name="市场60介于0_199", market20_min=0, market60_min=0, market60_max=199),
            MarketStateConfig(name="市场20<=399+市场60>=100", market20_min=0, market20_max=399, market60_min=100),
        ]
    )

    for threshold20, threshold60 in [
        (100, 200),
        (100, 300),
        (200, 200),
        (200, 300),
        (200, 400),
        (300, 300),
        (300, 400),
        (400, 400),
    ]:
        configs.append(
            MarketStateConfig(
                name=f"市场20>={threshold20}+市场60>={threshold60}",
                market20_min=threshold20,
                market60_min=threshold60,
            )
        )
    return configs


def format_market_state_config(config: MarketStateConfig) -> str:
    parts: list[str] = []
    if config.market20_min is not None:
        parts.append(f"市场20日高低差>={int(config.market20_min)}")
    if config.market20_max is not None:
        parts.append(f"市场20日高低差<={int(config.market20_max)}")
    if config.market60_min is not None:
        parts.append(f"市场60日高低差>={int(config.market60_min)}")
    if config.market60_max is not None:
        parts.append(f"市场60日高低差<={int(config.market60_max)}")
    return "; ".join(parts)


def build_candidate_book_for_market_state(strategy_mod, signal_df: pd.DataFrame, breadth_df: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    base = strategy_mod.AUCTION_RATIO_RANK_CONFIG
    candidates = signal_df.copy()
    if not breadth_df.empty:
        candidates = candidates.merge(breadth_df, on="日期", how="left")

    if base.named_conditions:
        candidates = strategy_mod.apply_named_conditions(candidates, base.named_conditions)
    if base.buy_filters:
        candidates = strategy_mod.apply_filter_rules(candidates, base.buy_filters)

    if config.market20_min is not None and "市场20日高低差" in candidates.columns:
        candidates = candidates[candidates["市场20日高低差"] >= config.market20_min].copy()
    if config.market20_max is not None and "市场20日高低差" in candidates.columns:
        candidates = candidates[candidates["市场20日高低差"] <= config.market20_max].copy()
    if config.market60_min is not None and "市场60日高低差" in candidates.columns:
        candidates = candidates[candidates["市场60日高低差"] >= config.market60_min].copy()
    if config.market60_max is not None and "市场60日高低差" in candidates.columns:
        candidates = candidates[candidates["市场60日高低差"] <= config.market60_max].copy()

    sort_columns = ["日期"]
    ascending = [True]
    for column, is_ascending in base.sort_by or ():
        sort_columns.append(column)
        ascending.append(is_ascending)
    if "基础代码" not in sort_columns:
        sort_columns.append("基础代码")
        ascending.append(True)

    candidates = candidates.sort_values(sort_columns, ascending=ascending, kind="stable")
    candidate_book = candidates.groupby("日期", group_keys=False).head(strategy_mod.MAX_POSITIONS).copy()
    candidate_book.reset_index(drop=True, inplace=True)
    return candidate_book


def analyze_breadth_buckets(candidate_book: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_book.empty or trade_df.empty:
        return pd.DataFrame()

    entry_df = candidate_book[
        ["日期", "股票代码", "市场20日高低差", "市场60日高低差", "市场20日新高数", "市场20日新低数"]
    ].copy()
    entry_df["买入日期"] = pd.to_datetime(entry_df["日期"]).dt.strftime("%Y%m%d")

    merged = trade_df.merge(
        entry_df.drop(columns=["日期"]),
        on=["买入日期", "股票代码"],
        how="left",
    )
    merged["市场20分组"] = pd.cut(
        merged["市场20日高低差"],
        bins=[-1, 99, 199, 299, 399, 999999],
        labels=["0-99", "100-199", "200-299", "300-399", "400+"],
    )
    merged["市场60分组"] = pd.cut(
        merged["市场60日高低差"],
        bins=[-999999, 99, 199, 299, 399, 999999],
        labels=["<=99", "100-199", "200-299", "300-399", "400+"],
    )

    group20 = (
        merged.groupby("市场20分组", observed=False)["单笔净收益率"]
        .agg(["count", "mean", "median"])
        .round(6)
        .reset_index()
        .rename(columns={"市场20分组": "宽度分组"})
    )
    group20.insert(0, "分组类型", "市场20日高低差")

    group60 = (
        merged.groupby("市场60分组", observed=False)["单笔净收益率"]
        .agg(["count", "mean", "median"])
        .round(6)
        .reset_index()
        .rename(columns={"市场60分组": "宽度分组"})
    )
    group60.insert(0, "分组类型", "市场60日高低差")

    return pd.concat([group20, group60], ignore_index=True)


def export_results(strategy_mod, result_df: pd.DataFrame, bucket_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-市场状态实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")
    if not bucket_df.empty:
        bucket_df.to_csv(f"{output_prefix}-买入日宽度分组.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    strategy_mod = load_strategy_module()
    signal_df = strategy_mod.load_factor_signal_data(strategy_mod.FACTOR_INPUT_CSV)
    breadth_df = load_breadth_data(strategy_mod)

    sector_columns = [column for column in signal_df.columns if any(keyword in str(column) for keyword in ["行业", "概念", "板块", "题材"])]

    all_codes = sorted(signal_df["基础代码"].dropna().astype(str).unique().tolist())
    histories = strategy_mod.fetch_histories(all_codes)
    configs = build_market_state_configs()

    rows: list[dict[str, object]] = []
    baseline_candidate_book = pd.DataFrame()
    baseline_trade_df = pd.DataFrame()

    for config in configs:
        candidate_book = build_candidate_book_for_market_state(strategy_mod, signal_df, breadth_df, config)
        if candidate_book.empty:
            row = {
                "策略名": config.name,
                "市场过滤": format_market_state_config(config),
                "期末净值": float("nan"),
                "总收益率": float("nan"),
                "最大回撤": float("nan"),
                "交易笔数": 0,
                "胜率": float("nan"),
                "收益回撤比": float("nan"),
                "候选池行数": 0,
                "候选交易日": 0,
            }
            rows.append(row)
            continue

        trade_df, _equity_df, summary_df = strategy_mod.run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        row["策略名"] = config.name
        row["市场过滤"] = format_market_state_config(config)
        row["候选池行数"] = int(len(candidate_book))
        row["候选交易日"] = int(candidate_book["日期"].nunique()) if not candidate_book.empty else 0
        rows.append(row)

        if config.name == "基线":
            baseline_candidate_book = candidate_book
            baseline_trade_df = trade_df

    result_df = pd.DataFrame(rows).sort_values(["收益回撤比", "总收益率"], ascending=[False, False]).reset_index(drop=True)
    bucket_df = analyze_breadth_buckets(baseline_candidate_book, baseline_trade_df)
    export_results(strategy_mod, result_df, bucket_df)

    print("市场状态实验完成")
    print(f"板块/行业相关字段数: {len(sector_columns)}")
    if sector_columns:
        print("可用板块字段:")
        for column in sector_columns:
            print(column)
    else:
        print("当前扩展因子文件没有行业/概念/板块字段，暂时无法做历史板块联动回测。")

    print("\n市场状态过滤结果:")
    print(
        result_df[
            [
                "策略名",
                "市场过滤",
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

    if not bucket_df.empty:
        print("\n基线买入日宽度分组:")
        print(bucket_df.to_string(index=False))


if __name__ == "__main__":
    main()
