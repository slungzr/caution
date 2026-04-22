from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
STOCK_ENHANCER_PATH = BASE_DIR / "竞价行业开盘个股增强探索.py"
SIZING_MODULE_PATH = BASE_DIR / "竞价仓位探索.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_best_candidate_book(strategy_mod, stock_mod):
    open_mod = stock_mod.load_open_sector_module()
    open_base_df, _ = stock_mod.prepare_open_base_df(strategy_mod, open_mod)
    best_config = next(config for config in stock_mod.build_configs(strategy_mod) if config.name == "量比优先排序")
    _, candidate_book = stock_mod.build_candidate_book(strategy_mod, open_base_df, best_config)
    return open_base_df, candidate_book


def numeric_score(record: dict[str, object], column: str) -> float:
    value = pd.to_numeric(pd.Series([record.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return math.nan
    return float(value)


def allocate_sqrt_by_column(records: list[dict[str, object]], cash: float, column: str):
    if not records:
        return []
    ranked_records = sorted(
        records,
        key=lambda record: numeric_score(record, column) if not math.isnan(numeric_score(record, column)) else -math.inf,
        reverse=True,
    )
    weights = []
    for record in ranked_records:
        score = numeric_score(record, column)
        weights.append(math.sqrt(score) if not math.isnan(score) and score > 0 else 0.0)
    total_weight = sum(weights)
    if total_weight <= 0:
        return []
    return [(record, cash * weight / total_weight) for record, weight in zip(ranked_records, weights) if weight > 0]


def build_policies(sizing_mod):
    return [
        sizing_mod.SizingPolicy("等权基线", sizing_mod.allocate_equal),
        sizing_mod.SizingPolicy(
            "量比50_30_20",
            lambda records, cash: sizing_mod.allocate_fixed_by_column(
                records, cash, column="量比", descending=True, weights=(0.5, 0.3, 0.2)
            ),
        ),
        sizing_mod.SizingPolicy(
            "量比60_30_10",
            lambda records, cash: sizing_mod.allocate_fixed_by_column(
                records, cash, column="量比", descending=True, weights=(0.6, 0.3, 0.1)
            ),
        ),
        sizing_mod.SizingPolicy(
            "量比平方根比例",
            lambda records, cash: allocate_sqrt_by_column(records, cash, "量比"),
        ),
        sizing_mod.SizingPolicy(
            "竞昨比50_30_20",
            lambda records, cash: sizing_mod.allocate_fixed_by_column(
                records, cash, column="竞昨成交比估算", descending=True, weights=(0.5, 0.3, 0.2)
            ),
        ),
        sizing_mod.SizingPolicy("竞昨比平方根比例", sizing_mod.allocate_sqrt_ratio),
        sizing_mod.SizingPolicy(
            "量比竞昨比混合平方根",
            lambda records, cash: allocate_mixed_sqrt(records, cash),
        ),
    ]


def allocate_mixed_sqrt(records: list[dict[str, object]], cash: float):
    if not records:
        return []
    ranked_records = sorted(
        records,
        key=lambda record: (
            numeric_score(record, "量比") if not math.isnan(numeric_score(record, "量比")) else -math.inf,
            numeric_score(record, "竞昨成交比估算") if not math.isnan(numeric_score(record, "竞昨成交比估算")) else -math.inf,
        ),
        reverse=True,
    )
    weights = []
    for record in ranked_records:
        volume_ratio = numeric_score(record, "量比")
        auction_ratio = numeric_score(record, "竞昨成交比估算")
        if math.isnan(volume_ratio) or volume_ratio <= 0 or math.isnan(auction_ratio) or auction_ratio <= 0:
            weights.append(0.0)
            continue
        weights.append(math.sqrt(volume_ratio) * math.sqrt(auction_ratio))
    total_weight = sum(weights)
    if total_weight <= 0:
        return []
    return [(record, cash * weight / total_weight) for record, weight in zip(ranked_records, weights) if weight > 0]


def main() -> None:
    strategy_mod = load_module("auction_backtest_open_stock_sizing", BACKTEST_MODULE_PATH)
    stock_mod = load_module("stock_enhancer_sizing", STOCK_ENHANCER_PATH)
    sizing_mod = load_module("sizing_module_open_stock", SIZING_MODULE_PATH)

    open_base_df, candidate_book = build_best_candidate_book(strategy_mod, stock_mod)
    histories = strategy_mod.fetch_histories(
        sorted(open_base_df["基础代码"].dropna().astype(str).unique().tolist())
    )

    rows: list[pd.DataFrame] = []
    best_name = None
    best_return = float("-inf")
    best_ratio = float("-inf")
    best_bundle = None

    for policy in build_policies(sizing_mod):
        trade_df, equity_df, summary_df = sizing_mod.run_sizing_policy_backtest(candidate_book, histories, strategy_mod, policy)
        row = summary_df.copy()
        row.insert(0, "候选策略", "量比优先排序")
        row.insert(1, "候选交易日", candidate_book["日期"].nunique())
        rows.append(row)

        total_return = float(row.iloc[0]["总收益率"])
        ratio = float(row.iloc[0]["收益回撤比"])
        if (total_return > best_return) or (
            math.isclose(total_return, best_return, rel_tol=1e-12, abs_tol=1e-12) and ratio > best_ratio
        ):
            best_return = total_return
            best_ratio = ratio
            best_name = policy.name
            best_bundle = (trade_df, equity_df, summary_df)

    result_df = pd.concat(rows, ignore_index=True)
    result_df = result_df.sort_values(["总收益率", "收益回撤比"], ascending=[False, False]).reset_index(drop=True)
    output_prefix = f"{strategy_mod.OUTPUT_STEM}-行业开盘个股分仓实验"
    result_df.to_csv(f"{output_prefix}-结果.csv", index=False, encoding="utf-8-sig")

    if best_bundle is not None:
        trade_df, equity_df, summary_df = best_bundle
        trade_df.to_csv(f"{output_prefix}-最佳策略交易明细.csv", index=False, encoding="utf-8-sig")
        export_equity_df = equity_df.copy()
        export_equity_df["日期"] = export_equity_df["日期"].dt.strftime("%Y%m%d")
        export_equity_df.to_csv(f"{output_prefix}-最佳策略净值.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(f"{output_prefix}-最佳策略摘要.csv", index=False, encoding="utf-8-sig")
        strategy_mod.save_plot(equity_df, Path(f"{output_prefix}-最佳策略净值.png"))
        pd.DataFrame([{"最佳策略": best_name}]).to_csv(
            f"{output_prefix}-最佳策略名称.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("行业开盘个股分仓实验完成")
    print(
        result_df[
            ["分仓策略", "期末净值", "总收益率", "最大回撤", "交易笔数", "胜率", "收益回撤比", "平均单笔净收益率"]
        ].to_string(index=False)
    )
    if best_name is not None:
        print(f"\n最佳分仓策略: {best_name}")


if __name__ == "__main__":
    main()
