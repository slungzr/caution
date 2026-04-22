from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
OPEN_SECTOR_MODULE_PATH = BASE_DIR / "竞价行业开盘联动探索.py"


@dataclass(frozen=True)
class EnhancedOpenSectorConfig:
    name: str
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
    return load_module("auction_backtest_open_enhanced", BACKTEST_MODULE_PATH)


def load_open_sector_module():
    return load_module("open_sector_enhanced", OPEN_SECTOR_MODULE_PATH)


def build_configs(strategy_mod) -> list[EnhancedOpenSectorConfig]:
    base_sort = strategy_mod.AUCTION_RATIO_RANK_CONFIG.sort_by or ()
    heat_column = strategy_mod.BUY_RANK_COLUMN
    return [
        EnhancedOpenSectorConfig(name="开盘行业基线", sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=-0.30%", open_change_min=-0.0030, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=-0.20%", open_change_min=-0.0020, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=-0.10%", open_change_min=-0.0010, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=-0.05%", open_change_min=-0.0005, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>0", open_change_min=0.0, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=0.05%", open_change_min=0.0005, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=0.10%", open_change_min=0.0010, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=0.15%", open_change_min=0.0015, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=0.20%", open_change_min=0.0020, sort_by=base_sort),
        EnhancedOpenSectorConfig(name="行业开盘涨幅>=0.30%", open_change_min=0.0030, sort_by=base_sort),
        EnhancedOpenSectorConfig(
            name="行业开盘涨幅优先排序",
            sort_by=(
                ("申万一级行业开盘涨幅", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        EnhancedOpenSectorConfig(
            name="行业开盘涨幅>0+行业优先排序",
            open_change_min=0.0,
            sort_by=(
                ("申万一级行业开盘涨幅", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        EnhancedOpenSectorConfig(
            name="行业开盘涨幅>=-0.10%+行业优先排序",
            open_change_min=-0.0010,
            sort_by=(
                ("申万一级行业开盘涨幅", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        EnhancedOpenSectorConfig(
            name="行业开盘涨幅排名优先排序",
            sort_by=(
                ("申万一级行业开盘涨幅排名", True),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        EnhancedOpenSectorConfig(
            name="行业开盘涨幅>0+排名优先排序",
            open_change_min=0.0,
            sort_by=(
                ("申万一级行业开盘涨幅排名", True),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
    ]


def build_candidate_book(strategy_mod, base_candidate_df: pd.DataFrame, config: EnhancedOpenSectorConfig) -> pd.DataFrame:
    candidates = base_candidate_df.copy()
    if config.open_change_min is not None:
        candidates = candidates[
            pd.to_numeric(candidates["申万一级行业开盘涨幅"], errors="coerce") >= config.open_change_min
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
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘增强实验"
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
    open_mod = load_open_sector_module()
    sector_mod = open_mod.load_sector_module()
    signal_df, breadth_df = open_mod.prepare_signal_df(strategy_mod, sector_mod)
    base_candidate_df = open_mod.build_base_candidate_df(strategy_mod, sector_mod, signal_df, breadth_df)
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
                    "行业开盘过滤": f"开盘涨幅>={config.open_change_min}",
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
        row["行业开盘过滤"] = f"开盘涨幅>={config.open_change_min}"
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
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘增强实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        export_best_bundle(strategy_mod, *best_bundle)

    print("行业开盘增强实验完成")
    print(
        result_df[
            ["策略名称", "期末净值", "总收益率", "最大回撤", "交易笔数", "胜率", "收益回撤比", "候选交易日"]
        ].to_string(index=False)
    )
    if best_bundle is not None:
        print(f"\n最佳增强策略: {best_bundle[0]}")


if __name__ == "__main__":
    main()
