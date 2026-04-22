from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BACKTEST_MODULE_PATH = BASE_DIR / "竞价爬升策略回测.py"
PLAN_FILE_GLOB = "今日操作清单-*.csv"
PLAN_JSON_TEMPLATE = "今日操作清单-{date_text}.json"
OUTPUT_PREFIX = BASE_DIR / "实时行业前向观察"


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("auction_backtest_forward", BACKTEST_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载回测模块: {BACKTEST_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_stock_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def load_plan_payload(plan_date: str) -> dict[str, Any]:
    payload_path = BASE_DIR / PLAN_JSON_TEMPLATE.format(date_text=plan_date)
    if not payload_path.exists():
        return {}
    return json.loads(payload_path.read_text(encoding="utf-8"))


def load_selected_records() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(BASE_DIR.glob(PLAN_FILE_GLOB)):
        date_match = csv_path.stem.split("-")[-1]
        if len(date_match) != 8 or not date_match.isdigit():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if df.empty:
            continue
        df["买入日期"] = date_match
        payload = load_plan_payload(date_match)
        status = payload.get("status", {})
        df["行业强度口径"] = status.get("行业强度口径", "")
        df["行业涨幅排名阈值"] = status.get("行业涨幅排名阈值")
        df["市场20日高低差"] = status.get("市场20日高低差")
        df["开仓开关"] = status.get("开仓开关")
        if "基础代码" not in df.columns:
            df["基础代码"] = df["股票代码"].apply(normalize_stock_code)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["买入日期"] = pd.to_datetime(merged_df["买入日期"], format="%Y%m%d", errors="coerce").dt.normalize()
    merged_df["基础代码"] = merged_df["基础代码"].astype(str).str.zfill(6)
    return merged_df.sort_values(["买入日期", "排序名次", "基础代码"]).reset_index(drop=True)


def evaluate_trade(strategy_mod, record: dict[str, Any], history_df: pd.DataFrame) -> dict[str, Any]:
    buy_date = pd.Timestamp(record["买入日期"]).normalize()
    open_price = float(record["开盘价:不复权今日"])
    allocated_cash = 1_000_000 / 3
    entry_exec_price = open_price * (1 + strategy_mod.SLIPPAGE_RATE)
    shares = allocated_cash / (entry_exec_price * (1 + strategy_mod.BUY_COMMISSION_RATE))
    last_close = math.nan
    last_trade_date = None

    for trade_date in history_df.index:
        trade_date = pd.Timestamp(trade_date).normalize()
        if trade_date <= buy_date:
            continue

        quote, exact_match = strategy_mod.get_quote_on_or_before(history_df, trade_date)
        if quote is None or not exact_match:
            continue
        if not pd.isna(quote.get("close")):
            last_close = float(quote["close"])
            last_trade_date = trade_date

        prev_close = float(quote.get("prev_close")) if not pd.isna(quote.get("prev_close")) else math.nan
        close_price = float(quote.get("close")) if not pd.isna(quote.get("close")) else math.nan
        if strategy_mod.is_limit_up_close(record["基础代码"], record["股票简称"], close_price, prev_close):
            continue

        exit_exec_price = close_price * (1 - strategy_mod.SLIPPAGE_RATE)
        net_proceeds = shares * exit_exec_price * (1 - strategy_mod.SELL_COMMISSION_RATE - strategy_mod.STAMP_DUTY_RATE)
        return {
            "观察状态": "已卖出",
            "卖出日期": trade_date.strftime("%Y%m%d"),
            "最后可用日期": trade_date.strftime("%Y%m%d"),
            "买入执行价": round(entry_exec_price, 4),
            "卖出执行价": round(exit_exec_price, 4),
            "单笔净收益率": round(net_proceeds / allocated_cash - 1, 6),
            "持有自然日": int((trade_date - buy_date).days),
            "最新收盘价": round(close_price, 4),
        }

    result = {
        "观察状态": "待卖出",
        "卖出日期": "",
        "最后可用日期": last_trade_date.strftime("%Y%m%d") if last_trade_date is not None else "",
        "买入执行价": round(entry_exec_price, 4),
        "卖出执行价": math.nan,
        "单笔净收益率": math.nan,
        "持有自然日": int((last_trade_date - buy_date).days) if last_trade_date is not None else 0,
        "最新收盘价": round(last_close, 4) if not math.isnan(last_close) else math.nan,
    }
    return result


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    closed_df = result_df[result_df["观察状态"] == "已卖出"].copy()
    summary = {
        "样本总数": int(len(result_df)),
        "已卖出样本数": int(len(closed_df)),
        "待卖出样本数": int((result_df["观察状态"] == "待卖出").sum()),
        "胜率": float((closed_df["单笔净收益率"] > 0).mean()) if not closed_df.empty else math.nan,
        "平均单笔净收益率": float(closed_df["单笔净收益率"].mean()) if not closed_df.empty else math.nan,
        "中位数单笔净收益率": float(closed_df["单笔净收益率"].median()) if not closed_df.empty else math.nan,
        "平均持有自然日": float(closed_df["持有自然日"].mean()) if not closed_df.empty else math.nan,
    }
    return pd.DataFrame([summary])


def main() -> None:
    strategy_mod = load_strategy_module()
    selected_df = load_selected_records()
    if selected_df.empty:
        print("未找到可用于前向观察的今日操作清单")
        return

    codes = sorted(selected_df["基础代码"].dropna().astype(str).unique().tolist())
    histories = strategy_mod.fetch_histories(codes)

    rows: list[dict[str, Any]] = []
    for record in selected_df.to_dict("records"):
        history_df = histories.get(record["基础代码"], pd.DataFrame())
        if history_df.empty:
            row = dict(record)
            row.update(
                {
                    "观察状态": "缺失日线",
                    "卖出日期": "",
                    "最后可用日期": "",
                    "买入执行价": math.nan,
                    "卖出执行价": math.nan,
                    "单笔净收益率": math.nan,
                    "持有自然日": math.nan,
                    "最新收盘价": math.nan,
                }
            )
            rows.append(row)
            continue

        row = dict(record)
        row.update(evaluate_trade(strategy_mod, record, history_df))
        row["买入日期"] = pd.Timestamp(record["买入日期"]).strftime("%Y%m%d")
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["买入日期", "排序名次", "基础代码"]).reset_index(drop=True)
    summary_df = summarize_results(result_df)

    result_df.to_csv(OUTPUT_PREFIX.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(BASE_DIR / "实时行业前向观察-汇总.csv", index=False, encoding="utf-8-sig")
    pending_df = result_df[result_df["观察状态"] == "待卖出"].copy()
    pending_df.to_csv(BASE_DIR / "实时行业前向观察-待卖出.csv", index=False, encoding="utf-8-sig")

    print("实时行业前向观察已更新")
    print(summary_df.to_string(index=False))
    if not pending_df.empty:
        print("\n待卖出样本:")
        print(
            pending_df[
                [
                    "买入日期",
                    "股票代码",
                    "股票简称",
                    "申万一级行业",
                    "申万一级行业涨跌幅排名",
                    "持有自然日",
                    "最后可用日期",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
