from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
OPEN_SECTOR_MODULE_PATH = BASE_DIR / "竞价行业开盘联动探索.py"


@dataclass(frozen=True)
class StockEnhancementConfig:
    name: str
    buy_filters: dict[str, dict[str, Any]] | None = None
    named_conditions: tuple[str, ...] = ()
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
    return load_module("auction_backtest_open_stock_enhanced", BACKTEST_MODULE_PATH)


def load_open_sector_module():
    return load_module("open_sector_stock_enhanced", OPEN_SECTOR_MODULE_PATH)


def prepare_open_base_df(strategy_mod, open_mod) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_mod = open_mod.load_sector_module()
    signal_df, breadth_df = open_mod.prepare_signal_df(strategy_mod, sector_mod)
    base_candidate_df = open_mod.build_base_candidate_df(strategy_mod, sector_mod, signal_df, breadth_df)
    base_candidate_df = base_candidate_df[
        pd.to_numeric(base_candidate_df["申万一级行业开盘涨幅"], errors="coerce") > 0
    ].copy()
    base_candidate_df["竞价涨幅绝对值"] = pd.to_numeric(base_candidate_df["竞价涨幅今日"], errors="coerce").abs()
    return base_candidate_df, breadth_df


def build_configs(strategy_mod) -> list[StockEnhancementConfig]:
    base_sort = strategy_mod.AUCTION_RATIO_RANK_CONFIG.sort_by or ()
    heat_column = strategy_mod.BUY_RANK_COLUMN
    return [
        StockEnhancementConfig(name="行业开盘>0基线", sort_by=base_sort),
        StockEnhancementConfig(name="昨量大于前量", named_conditions=("yesterday_volume_gt_prev",), sort_by=base_sort),
        StockEnhancementConfig(name="昨量前量比>=1.2", buy_filters={"昨日前日成交量比": {"min": 1.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="昨量前量比>=1.5", buy_filters={"昨日前日成交量比": {"min": 1.5}}, sort_by=base_sort),
        StockEnhancementConfig(name="量比>=1.0", buy_filters={"量比": {"min": 1.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="量比>=1.2", buy_filters={"量比": {"min": 1.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="量比>=1.5", buy_filters={"量比": {"min": 1.5}}, sort_by=base_sort),
        StockEnhancementConfig(name="量比>=2.0", buy_filters={"量比": {"min": 2.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="热度<=10", buy_filters={heat_column: {"max": 10}}, sort_by=base_sort),
        StockEnhancementConfig(name="热度<=20", buy_filters={heat_column: {"max": 20}}, sort_by=base_sort),
        StockEnhancementConfig(name="热度<=30", buy_filters={heat_column: {"max": 30}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅>=-3", buy_filters={"竞价涨幅今日": {"min": -3.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅>=-2", buy_filters={"竞价涨幅今日": {"min": -2.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅>=-1", buy_filters={"竞价涨幅今日": {"min": -1.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅>=0", buy_filters={"竞价涨幅今日": {"min": 0.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅<=2", buy_filters={"竞价涨幅今日": {"max": 2.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅<=1", buy_filters={"竞价涨幅今日": {"max": 1.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅<=0", buy_filters={"竞价涨幅今日": {"max": 0.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅[-3,2]", buy_filters={"竞价涨幅今日": {"min": -3.0, "max": 2.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅[-2,2]", buy_filters={"竞价涨幅今日": {"min": -2.0, "max": 2.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅[-1,1]", buy_filters={"竞价涨幅今日": {"min": -1.0, "max": 1.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="昨日连板<=0", buy_filters={"连续涨停天数昨日": {"max": 0}}, sort_by=base_sort),
        StockEnhancementConfig(name="昨日连板<=1", buy_filters={"连续涨停天数昨日": {"max": 1}}, sort_by=base_sort),
        StockEnhancementConfig(name="DDX>=0", buy_filters={"大单动向(ddx值)昨日": {"min": 0.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价强度>=8.0", buy_filters={"竞价强度": {"min": 8.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价强度>=8.3", buy_filters={"竞价强度": {"min": 8.3}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价换手率>=0.2", buy_filters={"竞价换手率_openapi": {"min": 0.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价换手率>=0.3", buy_filters={"竞价换手率_openapi": {"min": 0.3}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价量比>=0.8", buy_filters={"竞价量比_openapi": {"min": 0.8}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价量比>=1.0", buy_filters={"竞价量比_openapi": {"min": 1.0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅[-3,2]+量比>=1.2", buy_filters={"竞价涨幅今日": {"min": -3.0, "max": 2.0}, "量比": {"min": 1.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅[-2,2]+量比>=1.2", buy_filters={"竞价涨幅今日": {"min": -2.0, "max": 2.0}, "量比": {"min": 1.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅<=2+昨量比>=1.2", buy_filters={"竞价涨幅今日": {"max": 2.0}, "昨日前日成交量比": {"min": 1.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="量比>=1.2+昨量比>=1.2", buy_filters={"量比": {"min": 1.2}, "昨日前日成交量比": {"min": 1.2}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅<=2+昨日连板<=0", buy_filters={"竞价涨幅今日": {"max": 2.0}, "连续涨停天数昨日": {"max": 0}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅[-3,2]+热度<=30", buy_filters={"竞价涨幅今日": {"min": -3.0, "max": 2.0}, heat_column: {"max": 30}}, sort_by=base_sort),
        StockEnhancementConfig(name="竞价涨幅<=2+DDX>=0", buy_filters={"竞价涨幅今日": {"max": 2.0}, "大单动向(ddx值)昨日": {"min": 0.0}}, sort_by=base_sort),
        StockEnhancementConfig(
            name="量比优先排序",
            sort_by=(
                ("量比", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        StockEnhancementConfig(
            name="昨量比优先排序",
            sort_by=(
                ("昨日前日成交量比", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        StockEnhancementConfig(
            name="竞价涨幅温和优先排序",
            sort_by=(
                ("竞价涨幅绝对值", True),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        StockEnhancementConfig(
            name="竞价涨幅[-3,2]+量比优先排序",
            buy_filters={"竞价涨幅今日": {"min": -3.0, "max": 2.0}},
            sort_by=(
                ("量比", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
    ]


def build_candidate_book(strategy_mod, open_base_df: pd.DataFrame, config: StockEnhancementConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = open_base_df.copy()
    if config.named_conditions:
        candidates = strategy_mod.apply_named_conditions(candidates, config.named_conditions)
    if config.buy_filters:
        candidates = strategy_mod.apply_filter_rules(candidates, config.buy_filters)
    if candidates.empty:
        return candidates, candidates

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
    sorted_candidates = candidates.sort_values(sort_columns, ascending=ascending, kind="stable").copy()
    candidate_book = sorted_candidates.groupby("日期", group_keys=False).head(strategy_mod.MAX_POSITIONS).copy()
    candidate_book.reset_index(drop=True, inplace=True)
    return sorted_candidates, candidate_book


def export_best_bundle(strategy_mod, candidate_df: pd.DataFrame, trade_df: pd.DataFrame, equity_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘个股增强实验"
    keep_columns = [
        "日期",
        "股票代码",
        "股票简称",
        "申万一级行业代码",
        "申万一级行业",
        "申万一级行业开盘涨幅",
        "申万一级行业开盘涨幅排名",
        "竞昨成交比估算",
        "竞价涨幅今日",
        "量比",
        "昨日前日成交量比",
        "个股热度排名昨日",
        "大单动向(ddx值)昨日",
        "连续涨停天数昨日",
        "竞价匹配金额_openapi",
    ]
    keep_columns = [column for column in keep_columns if column in candidate_df.columns]
    export_candidate_df = candidate_df.copy()
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


def main() -> None:
    strategy_mod = load_strategy_module()
    open_mod = load_open_sector_module()
    open_base_df, _ = prepare_open_base_df(strategy_mod, open_mod)

    if open_base_df.empty:
        raise RuntimeError("行业开盘>0基线候选宇宙为空")

    histories = strategy_mod.fetch_histories(
        sorted(open_base_df["基础代码"].dropna().astype(str).unique().tolist())
    )

    rows: list[dict[str, object]] = []
    best_bundle: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    best_return = float("-inf")
    best_ratio = float("-inf")

    for config in build_configs(strategy_mod):
        filtered_candidates, candidate_book = build_candidate_book(strategy_mod, open_base_df, config)
        if candidate_book.empty:
            rows.append(
                {
                    "策略名称": config.name,
                    "期末净值": float("nan"),
                    "总收益率": float("nan"),
                    "最大回撤": float("nan"),
                    "交易笔数": 0,
                    "胜率": float("nan"),
                    "收益回撤比": float("nan"),
                    "候选池行数": 0,
                    "候选交易日": 0,
                    "基线宇宙覆盖率": 0.0,
                }
            )
            continue

        trade_df, equity_df, summary_df = strategy_mod.run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        row["策略名称"] = config.name
        row["候选池行数"] = int(len(filtered_candidates))
        row["候选交易日"] = int(candidate_book["日期"].nunique())
        row["基线宇宙覆盖率"] = round(len(filtered_candidates) / len(open_base_df), 6)
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
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘个股增强实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        export_best_bundle(strategy_mod, *best_bundle[1:])
        pd.DataFrame([{"最佳策略": best_bundle[0]}]).to_csv(
            f"{output_prefix}-最佳策略名称.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("行业开盘个股增强实验完成")
    print(
        result_df[
            ["策略名称", "期末净值", "总收益率", "最大回撤", "交易笔数", "胜率", "收益回撤比", "候选交易日", "基线宇宙覆盖率"]
        ].to_string(index=False)
    )
    if best_bundle is not None:
        print(f"\n最佳策略: {best_bundle[0]}")


if __name__ == "__main__":
    main()
