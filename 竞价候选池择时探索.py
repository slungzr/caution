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
class GateConfig:
    name: str
    min_candidate_count: int | None = None
    min_top1_unmatched: float | None = None
    min_top3_mean_unmatched: float | None = None
    min_top1_ratio: float | None = None
    min_top3_mean_ratio: float | None = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_strategy_module():
    return load_module("auction_backtest_pool_gate", BACKTEST_MODULE_PATH)


def load_open_sector_module():
    return load_module("open_sector_pool_gate", OPEN_SECTOR_MODULE_PATH)


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


def build_daily_candidate_book(strategy_mod, base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_columns = ["日期", "竞价未匹配占比", "竞昨成交比估算", strategy_mod.BUY_RANK_COLUMN, "基础代码"]
    ascending = [True, False, False, True, True]
    sorted_df = base_df.sort_values(sort_columns, ascending=ascending, kind="stable").copy()
    candidate_book = sorted_df.groupby("日期", group_keys=False).head(strategy_mod.MAX_POSITIONS).copy()
    candidate_book.reset_index(drop=True, inplace=True)
    return sorted_df, candidate_book


def build_day_stats(sorted_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, group in sorted_df.groupby("日期", sort=False):
        top3 = group.head(3)
        rows.append(
            {
                "日期": pd.Timestamp(trade_date),
                "候选数": int(len(group)),
                "top1未匹配占比": float(pd.to_numeric(top3["竞价未匹配占比"], errors="coerce").iloc[0]),
                "top1竞昨成交比": float(pd.to_numeric(top3["竞昨成交比估算"], errors="coerce").iloc[0]),
                "top3平均未匹配占比": float(pd.to_numeric(top3["竞价未匹配占比"], errors="coerce").mean()),
                "top3平均竞昨成交比": float(pd.to_numeric(top3["竞昨成交比估算"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("日期").reset_index(drop=True)


def build_configs() -> list[GateConfig]:
    configs = [GateConfig(name="尾段基线")]
    for threshold in [2, 3]:
        configs.append(GateConfig(name=f"候选数>={threshold}", min_candidate_count=threshold))
    for threshold in [0.01, 0.015, 0.02, 0.03]:
        configs.append(GateConfig(name=f"top1未匹配占比>={threshold:.3f}", min_top1_unmatched=threshold))
    for threshold in [0.005, 0.01, 0.015, 0.02]:
        configs.append(GateConfig(name=f"top3平均未匹配占比>={threshold:.3f}", min_top3_mean_unmatched=threshold))
    for threshold in [0.015, 0.02, 0.03]:
        configs.append(GateConfig(name=f"top1竞昨成交比>={threshold:.3f}", min_top1_ratio=threshold))
    for threshold in [0.012, 0.015, 0.02]:
        configs.append(GateConfig(name=f"top3平均竞昨成交比>={threshold:.3f}", min_top3_mean_ratio=threshold))
    configs.extend(
        [
            GateConfig(name="候选数>=2+top3平均未匹配占比>=0.010", min_candidate_count=2, min_top3_mean_unmatched=0.01),
            GateConfig(name="候选数>=2+top1竞昨成交比>=0.020", min_candidate_count=2, min_top1_ratio=0.02),
            GateConfig(name="top1未匹配占比>=0.010+top1竞昨成交比>=0.020", min_top1_unmatched=0.01, min_top1_ratio=0.02),
        ]
    )
    return configs


def apply_gate(day_stats_df: pd.DataFrame, config: GateConfig) -> set[pd.Timestamp]:
    mask = pd.Series(True, index=day_stats_df.index)
    if config.min_candidate_count is not None:
        mask &= day_stats_df["候选数"] >= config.min_candidate_count
    if config.min_top1_unmatched is not None:
        mask &= day_stats_df["top1未匹配占比"] >= config.min_top1_unmatched
    if config.min_top3_mean_unmatched is not None:
        mask &= day_stats_df["top3平均未匹配占比"] >= config.min_top3_mean_unmatched
    if config.min_top1_ratio is not None:
        mask &= day_stats_df["top1竞昨成交比"] >= config.min_top1_ratio
    if config.min_top3_mean_ratio is not None:
        mask &= day_stats_df["top3平均竞昨成交比"] >= config.min_top3_mean_ratio
    return set(day_stats_df.loc[mask, "日期"])


def main() -> None:
    strategy_mod = load_strategy_module()
    open_mod = load_open_sector_module()
    base_df = prepare_tail_base_df(strategy_mod, open_mod)
    sorted_df, candidate_book = build_daily_candidate_book(strategy_mod, base_df)
    day_stats_df = build_day_stats(sorted_df)
    histories = strategy_mod.fetch_histories(sorted(base_df["基础代码"].dropna().astype(str).unique().tolist()))

    rows: list[dict[str, object]] = []
    best_bundle = None
    best_return = float("-inf")
    best_ratio = float("-inf")

    for config in build_configs():
        allowed_dates = apply_gate(day_stats_df, config)
        gated_book = candidate_book[candidate_book["日期"].isin(allowed_dates)].copy()
        if gated_book.empty:
            rows.append(
                {
                    "策略名称": config.name,
                    "期末净值": float("nan"),
                    "总收益率": float("nan"),
                    "最大回撤": float("nan"),
                    "交易笔数": 0,
                    "胜率": float("nan"),
                    "收益回撤比": float("nan"),
                    "交易日数": 0,
                }
            )
            continue

        trade_df, equity_df, summary_df = strategy_mod.run_backtest(gated_book, histories)
        row = summary_df.iloc[0].to_dict()
        row["策略名称"] = config.name
        row["交易日数"] = int(gated_book["日期"].nunique())
        rows.append(row)

        total_return = float(row["总收益率"])
        ratio = float(row["收益回撤比"])
        if (total_return > best_return) or (
            math.isclose(total_return, best_return, rel_tol=1e-12, abs_tol=1e-12) and ratio > best_ratio
        ):
            best_return = total_return
            best_ratio = ratio
            best_bundle = (config.name, gated_book, trade_df, equity_df, summary_df)

    result_df = pd.DataFrame(rows).sort_values(["总收益率", "收益回撤比"], ascending=[False, False]).reset_index(drop=True)
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-候选池择时实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        label, gated_book, trade_df, equity_df, summary_df = best_bundle
        export_candidate_df = gated_book.copy()
        export_candidate_df["日期"] = pd.to_datetime(export_candidate_df["日期"]).dt.strftime("%Y%m%d")
        export_candidate_df.to_csv(f"{output_prefix}-最佳策略候选池.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
        export_equity_df = equity_df.copy()
        export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
        export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")
        strategy_mod.save_plot(equity_df, Path(f"{output_prefix}-最佳策略净值.png"))
        pd.DataFrame([{"最佳策略": label}]).to_csv(f"{output_prefix}-最佳策略名称.csv", index=False, encoding="utf-8-sig")

    print("候选池择时实验完成")
    print(result_df[["策略名称", "总收益率", "最大回撤", "交易笔数", "收益回撤比", "交易日数"]].to_string(index=False))
    if best_bundle is not None:
        print(f"\n最佳策略: {best_bundle[0]}")


if __name__ == "__main__":
    main()
