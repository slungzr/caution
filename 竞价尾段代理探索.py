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
class TailProxyConfig:
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


def prepare_base_df(strategy_mod, open_mod) -> pd.DataFrame:
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


def build_configs(strategy_mod) -> list[TailProxyConfig]:
    base_sort = strategy_mod.AUCTION_RATIO_RANK_CONFIG.sort_by or ()
    heat_column = strategy_mod.BUY_RANK_COLUMN
    return [
        TailProxyConfig(name="行业开盘>0基线", sort_by=base_sort),
        TailProxyConfig(name="未匹配金额>=0", buy_filters={"竞价未匹配金额": {"min": 0.0}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配金额>=20万", buy_filters={"竞价未匹配金额": {"min": 200_000}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配金额>=50万", buy_filters={"竞价未匹配金额": {"min": 500_000}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配金额>=100万", buy_filters={"竞价未匹配金额": {"min": 1_000_000}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配占比>=0", buy_filters={"竞价未匹配占比": {"min": 0.0}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配占比>=1%", buy_filters={"竞价未匹配占比": {"min": 0.01}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配占比>=2%", buy_filters={"竞价未匹配占比": {"min": 0.02}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配占比>=3%", buy_filters={"竞价未匹配占比": {"min": 0.03}}, sort_by=base_sort),
        TailProxyConfig(name="未匹配占比>=5%", buy_filters={"竞价未匹配占比": {"min": 0.05}}, sort_by=base_sort),
        TailProxyConfig(
            name="未匹配占比优先排序",
            sort_by=(
                ("竞价未匹配占比", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        TailProxyConfig(
            name="未匹配金额优先排序",
            sort_by=(
                ("竞价未匹配金额", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
        TailProxyConfig(
            name="未匹配占比>=1%+优先排序",
            buy_filters={"竞价未匹配占比": {"min": 0.01}},
            sort_by=(
                ("竞价未匹配占比", False),
                ("竞昨成交比估算", False),
                (heat_column, True),
                ("基础代码", True),
            ),
        ),
    ]


def build_candidate_book(strategy_mod, base_df: pd.DataFrame, config: TailProxyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = base_df.copy()
    if config.buy_filters:
        filtered = strategy_mod.apply_filter_rules(filtered, config.buy_filters)
    if filtered.empty:
        return filtered, filtered

    sort_columns = ["日期"]
    ascending = [True]
    for column, is_ascending in config.sort_by or ():
        if column not in filtered.columns:
            raise KeyError(f"排序列不存在: {column}")
        sort_columns.append(column)
        ascending.append(is_ascending)
    if "基础代码" not in sort_columns:
        sort_columns.append("基础代码")
        ascending.append(True)

    filtered = filtered.sort_values(sort_columns, ascending=ascending, kind="stable").copy()
    candidate_book = filtered.groupby("日期", group_keys=False).head(strategy_mod.MAX_POSITIONS).copy()
    candidate_book.reset_index(drop=True, inplace=True)
    return filtered, candidate_book


def main() -> None:
    strategy_mod = load_module("tail_proxy_backtest", BACKTEST_MODULE_PATH)
    open_mod = load_module("tail_proxy_open_sector", OPEN_SECTOR_MODULE_PATH)
    base_df = prepare_base_df(strategy_mod, open_mod)
    histories = strategy_mod.fetch_histories(
        sorted(base_df["基础代码"].dropna().astype(str).unique().tolist())
    )

    rows: list[dict[str, object]] = []
    best_bundle = None
    best_return = float("-inf")
    best_ratio = float("-inf")

    for config in build_configs(strategy_mod):
        filtered_df, candidate_book = build_candidate_book(strategy_mod, base_df, config)
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
        row["候选池行数"] = int(len(filtered_df))
        row["候选交易日"] = int(candidate_book["日期"].nunique())
        row["基线宇宙覆盖率"] = round(len(filtered_df) / len(base_df), 6)
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
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-尾段代理实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        best_name, candidate_book, trade_df, equity_df, summary_df = best_bundle
        export_candidate = candidate_book.copy()
        export_candidate["日期"] = pd.to_datetime(export_candidate["日期"]).dt.strftime("%Y%m%d")
        keep_columns = [
            "日期",
            "股票代码",
            "股票简称",
            "申万一级行业",
            "申万一级行业开盘涨幅",
            "竞价匹配金额_openapi",
            "竞价未匹配金额",
            "竞价未匹配占比",
            "竞昨成交比估算",
            "个股热度排名昨日",
        ]
        keep_columns = [column for column in keep_columns if column in export_candidate.columns]
        export_candidate[keep_columns].to_csv(f"{output_prefix}-最佳策略候选池.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
        export_equity = equity_df.copy()
        export_equity["日期"] = export_equity["日期"].dt.strftime("%Y%m%d")
        export_equity.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")
        strategy_mod.save_plot(equity_df, Path(f"{output_prefix}-最佳策略净值.png"))
        pd.DataFrame([{"最佳策略": best_name}]).to_csv(f"{output_prefix}-最佳策略名称.csv", index=False, encoding="utf-8-sig")

    print("尾段代理实验完成")
    print(
        result_df[
            ["策略名称", "期末净值", "总收益率", "最大回撤", "交易笔数", "胜率", "收益回撤比", "候选交易日", "基线宇宙覆盖率"]
        ].to_string(index=False)
    )
    if best_bundle is not None:
        print(f"\n最佳策略: {best_bundle[0]}")


if __name__ == "__main__":
    main()
