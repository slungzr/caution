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
SMALL_BODY_THRESHOLD = 3.0


@dataclass(frozen=True)
class BodyConfig:
    name: str
    buy_filters: dict[str, dict[str, Any]] | None = None
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
    return load_module("auction_backtest_body", BACKTEST_MODULE_PATH)


def load_open_sector_module():
    return load_module("open_sector_body", OPEN_SECTOR_MODULE_PATH)


def prepare_tail_base_df(strategy_mod, open_mod) -> pd.DataFrame:
    sector_mod = open_mod.load_sector_module()
    signal_df, breadth_df = open_mod.prepare_signal_df(strategy_mod, sector_mod)
    base_candidate_df = open_mod.build_base_candidate_df(strategy_mod, sector_mod, signal_df, breadth_df)
    base_candidate_df = base_candidate_df[
        pd.to_numeric(base_candidate_df["申万一级行业开盘涨幅"], errors="coerce") > 0
    ].copy()
    base_candidate_df["竞价未匹配占比"] = (
        pd.to_numeric(base_candidate_df["竞价未匹配金额"], errors="coerce")
        / pd.to_numeric(base_candidate_df["竞价匹配金额_openapi"], errors="coerce")
    )
    return base_candidate_df


def build_configs(strategy_mod) -> list[BodyConfig]:
    heat_column = strategy_mod.BUY_RANK_COLUMN
    tail_sort = (
        ("竞价未匹配占比", False),
        ("竞昨成交比估算", False),
        (heat_column, True),
        ("基础代码", True),
    )
    y_col = "实体涨跌幅昨日"
    p_col = "实体涨跌幅前日"
    t = SMALL_BODY_THRESHOLD
    return [
        BodyConfig(name="尾段基线", sort_by=tail_sort),
        BodyConfig(name="昨日阴线", buy_filters={y_col: {"max": 0.0}}, sort_by=tail_sort),
        BodyConfig(name="昨日阳线", buy_filters={y_col: {"min": 0.0}}, sort_by=tail_sort),
        BodyConfig(name="昨日小阴线", buy_filters={y_col: {"min": -t, "max": 0.0}}, sort_by=tail_sort),
        BodyConfig(name="昨日大阴线", buy_filters={y_col: {"max": -t}}, sort_by=tail_sort),
        BodyConfig(name="昨日小阳线", buy_filters={y_col: {"min": 0.0, "max": t}}, sort_by=tail_sort),
        BodyConfig(name="前日阴线", buy_filters={p_col: {"max": 0.0}}, sort_by=tail_sort),
        BodyConfig(name="前日阳线", buy_filters={p_col: {"min": 0.0}}, sort_by=tail_sort),
        BodyConfig(name="前日小阳线", buy_filters={p_col: {"min": 0.0, "max": t}}, sort_by=tail_sort),
        BodyConfig(name="前日大阳线", buy_filters={p_col: {"min": t}}, sort_by=tail_sort),
        BodyConfig(
            name="昨日阴线+前日阳线",
            buy_filters={y_col: {"max": 0.0}, p_col: {"min": 0.0}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日小阴线+前日阳线",
            buy_filters={y_col: {"min": -t, "max": 0.0}, p_col: {"min": 0.0}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日阴线+前日小阳线",
            buy_filters={y_col: {"max": 0.0}, p_col: {"min": 0.0, "max": t}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日阴线+前日大阳线",
            buy_filters={y_col: {"max": 0.0}, p_col: {"min": t}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日小阴线+前日小阳线",
            buy_filters={y_col: {"min": -t, "max": 0.0}, p_col: {"min": 0.0, "max": t}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日小阴线+前日大阳线",
            buy_filters={y_col: {"min": -t, "max": 0.0}, p_col: {"min": t}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日阳线+前日大阳线",
            buy_filters={y_col: {"min": 0.0}, p_col: {"min": t}},
            sort_by=tail_sort,
        ),
        BodyConfig(
            name="昨日小阳线+前日大阳线",
            buy_filters={y_col: {"min": 0.0, "max": t}, p_col: {"min": t}},
            sort_by=tail_sort,
        ),
    ]


def build_candidate_book(strategy_mod, base_df: pd.DataFrame, config: BodyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = base_df.copy()
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


def summarize_validation(
    strategy_mod,
    candidate_book: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    split_date: pd.Timestamp,
) -> dict[str, object]:
    valid_candidate_book = candidate_book[candidate_book["日期"] >= split_date].copy()
    if valid_candidate_book.empty:
        return {
            "验证期总收益率": float("nan"),
            "验证期最大回撤": float("nan"),
            "验证期交易笔数": 0,
            "验证期收益回撤比": float("nan"),
            "验证期交易日": 0,
        }

    _, _, valid_summary_df = strategy_mod.run_backtest(valid_candidate_book, histories)
    row = valid_summary_df.iloc[0]
    return {
        "验证期总收益率": float(row["总收益率"]),
        "验证期最大回撤": float(row["最大回撤"]),
        "验证期交易笔数": int(row["交易笔数"]),
        "验证期收益回撤比": float(row["收益回撤比"]),
        "验证期交易日": int(valid_candidate_book["日期"].nunique()),
    }


def export_best_bundle(
    strategy_mod,
    label: str,
    candidate_book: pd.DataFrame,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-实体形态实验"
    export_candidate_df = candidate_book.copy()
    export_candidate_df["日期"] = pd.to_datetime(export_candidate_df["日期"]).dt.strftime("%Y%m%d")
    keep_columns = [
        "日期",
        "股票代码",
        "股票简称",
        "申万一级行业",
        "申万一级行业开盘涨幅",
        "竞价未匹配占比",
        "竞昨成交比估算",
        "个股热度排名昨日",
        "实体涨跌幅昨日",
        "实体涨跌幅前日",
    ]
    keep_columns = [column for column in keep_columns if column in export_candidate_df.columns]
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
    base_df = prepare_tail_base_df(strategy_mod, open_mod)
    if base_df.empty:
        raise RuntimeError("实体形态实验的候选宇宙为空")

    _, _, split_date = strategy_mod.split_signal_data_by_ratio(base_df[["日期"]].copy(), strategy_mod.TRAIN_RATIO)
    histories = strategy_mod.fetch_histories(sorted(base_df["基础代码"].dropna().astype(str).unique().tolist()))

    rows: list[dict[str, object]] = []
    best_bundle: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    best_return = float("-inf")
    best_valid_return = float("-inf")
    baseline_row: dict[str, object] | None = None

    for config in build_configs(strategy_mod):
        filtered_df, candidate_book = build_candidate_book(strategy_mod, base_df, config)
        if candidate_book.empty:
            row = {
                "策略名称": config.name,
                "总收益率": float("nan"),
                "最大回撤": float("nan"),
                "交易笔数": 0,
                "收益回撤比": float("nan"),
                "候选交易日": 0,
                "候选覆盖率": 0.0,
                "验证期总收益率": float("nan"),
                "验证期最大回撤": float("nan"),
                "验证期交易笔数": 0,
                "验证期收益回撤比": float("nan"),
                "验证期交易日": 0,
            }
            rows.append(row)
            continue

        trade_df, equity_df, summary_df = strategy_mod.run_backtest(candidate_book, histories)
        row = summary_df.iloc[0].to_dict()
        row["策略名称"] = config.name
        row["候选交易日"] = int(candidate_book["日期"].nunique())
        row["候选覆盖率"] = round(len(filtered_df) / len(base_df), 6)
        row.update(summarize_validation(strategy_mod, candidate_book, histories, split_date))
        rows.append(row)

        if config.name == "尾段基线":
            baseline_row = row.copy()

        total_return = float(row["总收益率"])
        valid_return = float(row["验证期总收益率"]) if not pd.isna(row["验证期总收益率"]) else float("-inf")
        if (total_return > best_return) or (
            math.isclose(total_return, best_return, rel_tol=1e-12, abs_tol=1e-12) and valid_return > best_valid_return
        ):
            best_return = total_return
            best_valid_return = valid_return
            best_bundle = (config.name, candidate_book, trade_df, equity_df, summary_df)

    result_df = pd.DataFrame(rows)
    if baseline_row is not None:
        result_df["总收益率增量"] = result_df["总收益率"] - float(baseline_row["总收益率"])
        result_df["验证期总收益率增量"] = result_df["验证期总收益率"] - float(baseline_row["验证期总收益率"])
    result_df = result_df.sort_values(
        ["总收益率", "验证期总收益率", "收益回撤比"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    output_prefix = f"{strategy_mod.OUTPUT_STEM}-实体形态实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        export_best_bundle(strategy_mod, *best_bundle)

    print("实体形态实验完成")
    print(f"小阴/小阳阈值: {SMALL_BODY_THRESHOLD:.1f}%")
    print(f"训练期截止前一日: {(split_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
    print(f"验证期起始日: {split_date.strftime('%Y-%m-%d')}")
    print(
        result_df[
            [
                "策略名称",
                "总收益率",
                "验证期总收益率",
                "最大回撤",
                "交易笔数",
                "收益回撤比",
                "总收益率增量",
                "验证期总收益率增量",
                "候选交易日",
                "候选覆盖率",
            ]
        ].to_string(index=False)
    )
    if best_bundle is not None:
        print(f"\n最佳策略: {best_bundle[0]}")


if __name__ == "__main__":
    main()
