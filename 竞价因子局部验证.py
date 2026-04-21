from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FACTOR_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
SOURCE_SCRIPT = BASE_DIR / "竞价爬升策略回测.py"
OUTPUT_STEM = BASE_DIR / "竞价爬升-20240504-局部验证"


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("auction_backtest", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_factor_data(csv_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"读取失败: {csv_path}")

    df["日期"] = pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce")
    df["基础代码"] = df["基础代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    factor_mask = df["竞价强度"].notna()
    df = df.loc[factor_mask].copy()

    numeric_columns = [
        "开盘价:不复权今日",
        "收盘价:不复权今日",
        "个股热度排名昨日",
        "竞价强度",
        "竞价匹配金额_openapi",
        "竞价未匹配金额",
        "竞价换手率_openapi",
        "竞价量比_openapi",
        "连续涨停天数_openapi",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def safe_name(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '_', value)


def build_local_configs(module):
    return [
        module.StrategyConfig(name="局部基线"),
        module.StrategyConfig(name="竞价强度>=8.5", buy_filters={"竞价强度": {"min": 8.5}}),
        module.StrategyConfig(
            name="竞价匹配金额>=5000万",
            buy_filters={"竞价匹配金额_openapi": {"min": 50_000_000}},
        ),
        module.StrategyConfig(
            name="竞价强度>=8.5+匹配金额>=5000万",
            buy_filters={"竞价强度": {"min": 8.5}, "竞价匹配金额_openapi": {"min": 50_000_000}},
        ),
        module.StrategyConfig(
            name="竞价强度>=8.3+匹配金额>=3000万",
            buy_filters={"竞价强度": {"min": 8.3}, "竞价匹配金额_openapi": {"min": 30_000_000}},
        ),
        module.StrategyConfig(
            name="竞价换手率>=0.3",
            buy_filters={"竞价换手率_openapi": {"min": 0.3}},
        ),
        module.StrategyConfig(
            name="金额>=5000万+昨日非跌停",
            buy_filters={
                "竞价匹配金额_openapi": {"min": 50_000_000},
                "涨跌幅:前复权昨日": {"min": -9.8},
            },
        ),
        module.StrategyConfig(
            name="金额>=5000万+实体昨日<前日",
            buy_filters={"竞价匹配金额_openapi": {"min": 50_000_000}},
            named_conditions=("yesterday_body_lt_prev",),
        ),
        module.StrategyConfig(
            name="金额>=5000万+热度<=25",
            buy_filters={
                "竞价匹配金额_openapi": {"min": 50_000_000},
                "个股热度排名昨日": {"max": 25},
            },
        ),
        module.StrategyConfig(
            name="金额>=5000万+非跌停+实体弱化",
            buy_filters={
                "竞价匹配金额_openapi": {"min": 50_000_000},
                "涨跌幅:前复权昨日": {"min": -9.8},
            },
            named_conditions=("yesterday_body_lt_prev",),
        ),
        module.StrategyConfig(
            name="金额>=5000万+非跌停+热度<=25",
            buy_filters={
                "竞价匹配金额_openapi": {"min": 50_000_000},
                "涨跌幅:前复权昨日": {"min": -9.8},
                "个股热度排名昨日": {"max": 25},
            },
        ),
        module.StrategyConfig(
            name="金额>=5000万+实体弱化+热度<=25",
            buy_filters={
                "竞价匹配金额_openapi": {"min": 50_000_000},
                "个股热度排名昨日": {"max": 25},
            },
            named_conditions=("yesterday_body_lt_prev",),
        ),
        module.StrategyConfig(
            name="金额>=5000万+三方向全开",
            buy_filters={
                "竞价匹配金额_openapi": {"min": 50_000_000},
                "涨跌幅:前复权昨日": {"min": -9.8},
                "个股热度排名昨日": {"max": 25},
            },
            named_conditions=("yesterday_body_lt_prev",),
        ),
    ]


def main() -> None:
    module = load_backtest_module()
    signal_df = load_factor_data(FACTOR_CSV)
    configs = build_local_configs(module)

    codes = sorted(signal_df["基础代码"].dropna().astype(str).unique().tolist())
    histories = module.fetch_histories(codes)

    results: list[pd.DataFrame] = []
    best_name = None
    best_score = None

    for config in configs:
        candidate_book = module.build_candidate_book_with_config(
            signal_df=signal_df,
            breadth_df=pd.DataFrame(),
            buy_filters=config.buy_filters,
            named_conditions=config.named_conditions,
            market_filter=config.market_filter,
            max_positions=module.MAX_POSITIONS,
        )
        if candidate_book.empty:
            row = pd.DataFrame(
                [{
                    "策略名": config.name,
                    "候选池行数": 0,
                    "候选交易日": 0,
                    "期末净值": None,
                    "总收益率": None,
                    "最大回撤": None,
                    "交易笔数": 0,
                    "胜率": None,
                    "平均单笔净收益率": None,
                    "收益回撤比": None,
                }]
            )
        else:
            trade_df, equity_df, summary_df = module.run_backtest(candidate_book, histories)
            row = summary_df.copy()
            row.insert(0, "策略名", config.name)
            row.insert(1, "候选池行数", len(candidate_book))
            row.insert(2, "候选交易日", candidate_book["日期"].nunique())
            file_label = safe_name(config.name)
            equity_export = equity_df.copy()
            equity_export["日期"] = equity_export["日期"].dt.strftime("%Y%m%d")
            equity_export.to_csv(f"{OUTPUT_STEM}-{file_label}-净值.csv", index=False, encoding="utf-8-sig")
            trade_df.to_csv(f"{OUTPUT_STEM}-{file_label}-交易明细.csv", index=False, encoding="utf-8-sig")

            score = row.iloc[0]["收益回撤比"]
            if pd.notna(score) and (best_score is None or score > best_score):
                best_score = score
                best_name = config.name

        results.append(row)

    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.sort_values(["收益回撤比", "总收益率"], ascending=[False, False], na_position="last")
    result_df.to_csv(f"{OUTPUT_STEM}-结果.csv", index=False, encoding="utf-8-sig")
    print(result_df.to_string(index=False))
    print(f"\n最佳策略: {best_name}")


if __name__ == "__main__":
    main()
