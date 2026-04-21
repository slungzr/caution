from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = BASE_DIR / "竞价爬升策略回测.py"
FACTOR_CSV = BASE_DIR / "竞价爬升-20240504-扩展因子-验证期.csv"
MARKET_BREADTH_CSV = BASE_DIR / "竞价爬升-20240504-市场宽度.csv"
OUTPUT_STEM = BASE_DIR / "竞价爬升-20240504-风控验证"


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("auction_backtest_risk", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_name(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '_', value)


def risk_multiplier(mode: str, equity_records: list[dict[str, float]]) -> float:
    if mode in {"baseline", "breadth_ge_0", "breadth_ge_300", "quality_strength_ge_8_5", "quality_count_ge_2_strength_ge_8_5"}:
        return 1.0

    if mode in {"two_down_pause", "two_down_half"}:
        if len(equity_records) < 3:
            return 1.0
        e1 = float(equity_records[-1]["总权益"])
        e2 = float(equity_records[-2]["总权益"])
        e3 = float(equity_records[-3]["总权益"])
        two_down = e1 < e2 and e2 < e3
        if not two_down:
            return 1.0
        return 0.0 if mode == "two_down_pause" else 0.5

    if mode == "drawdown_control":
        if not equity_records:
            return 1.0
        latest = equity_records[-1]
        drawdown = float(latest.get("回撤", 0.0)) if latest.get("回撤") is not None else 0.0
        if drawdown <= -0.12:
            return 0.0
        if drawdown <= -0.08:
            return 0.5
        return 1.0

    if mode == "drawdown_tight":
        if not equity_records:
            return 1.0
        latest = equity_records[-1]
        drawdown = float(latest.get("回撤", 0.0)) if latest.get("回撤") is not None else 0.0
        if drawdown <= -0.08:
            return 0.0
        if drawdown <= -0.05:
            return 0.5
        return 1.0

    raise ValueError(f"未知风控模式: {mode}")


def open_gate(mode: str, trade_date: pd.Timestamp, day_records: list[dict[str, float]], market_map: dict[pd.Timestamp, dict[str, float]]) -> bool:
    if mode in {"baseline", "two_down_pause", "two_down_half", "drawdown_control", "drawdown_tight"}:
        return True

    market_row = market_map.get(trade_date, {})
    market20 = market_row.get("市场20日高低差")

    if mode == "breadth_ge_0":
        return market20 is not None and not pd.isna(market20) and float(market20) >= 0

    if mode == "breadth_ge_300":
        return market20 is not None and not pd.isna(market20) and float(market20) >= 300

    numeric_strengths = [float(record.get("竞价强度")) for record in day_records if pd.notna(record.get("竞价强度"))]
    avg_strength = sum(numeric_strengths) / len(numeric_strengths) if numeric_strengths else math.nan

    if mode == "quality_strength_ge_8_5":
        return not pd.isna(avg_strength) and avg_strength >= 8.5

    if mode == "quality_count_ge_2_strength_ge_8_5":
        return len(day_records) >= 2 and not pd.isna(avg_strength) and avg_strength >= 8.5

    raise ValueError(f"未知开仓开关模式: {mode}")


def run_backtest_with_risk(
    module,
    candidate_book: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    mode: str,
    market_map: dict[pd.Timestamp, dict[str, float]],
):
    if candidate_book.empty:
        raise ValueError("候选池为空，无法回测")

    candidate_groups = {
        pd.Timestamp(trade_date): group.sort_values(module.BUY_RANK_COLUMN, kind="stable").to_dict("records")
        for trade_date, group in candidate_book.groupby("日期")
    }
    trading_calendar = module.build_trading_calendar(candidate_book, histories)
    trading_calendar = [trade_date for trade_date in trading_calendar if trade_date >= candidate_book["日期"].min()]

    cash = module.INITIAL_CAPITAL
    positions: list[module.Position] = []
    trade_records: list[dict[str, float]] = []
    equity_records: list[dict[str, float]] = []

    for trade_date in trading_calendar:
        multiplier = risk_multiplier(mode, equity_records)
        open_slots = module.MAX_POSITIONS - len(positions)

        if open_slots > 0 and cash > 1e-8 and multiplier > 0 and trade_date in candidate_groups and open_gate(mode, trade_date, candidate_groups[trade_date], market_map):
            buyable_records: list[dict[str, float]] = []
            for record in candidate_groups[trade_date]:
                if len(buyable_records) >= open_slots:
                    break
                history_df = histories.get(record["基础代码"], pd.DataFrame())
                quote, exact_match = module.get_quote_on_or_before(history_df, trade_date)
                if quote is None or not exact_match:
                    continue
                if pd.isna(record.get("开盘价:不复权今日")) or float(record["开盘价:不复权今日"]) <= 0:
                    continue
                buyable_records.append(record)

            if buyable_records:
                deploy_cash = cash * multiplier
                budget_per_trade = deploy_cash / len(buyable_records)
                if budget_per_trade > 0:
                    for record in buyable_records:
                        position = module.create_position(record, budget_per_trade)
                        cash -= budget_per_trade
                        positions.append(position)

        positions_to_close: list[tuple[object, float]] = []
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, exact_match = module.get_quote_on_or_before(history_df, trade_date)
            if quote is not None and not pd.isna(quote.get("close")):
                position.last_close = float(quote["close"])
            if trade_date <= position.entry_date or quote is None or not exact_match:
                continue
            prev_close = float(quote.get("prev_close")) if not pd.isna(quote.get("prev_close")) else math.nan
            close_price = float(quote.get("close")) if not pd.isna(quote.get("close")) else math.nan
            if not module.is_limit_up_close(position.code, position.name, close_price, prev_close):
                positions_to_close.append((position, close_price))

        for position, close_price in positions_to_close:
            exec_exit_price = close_price * (1 - module.SLIPPAGE_RATE)
            net_proceeds = position.shares * exec_exit_price * (1 - module.SELL_COMMISSION_RATE - module.STAMP_DUTY_RATE)
            cash += net_proceeds
            trade_records.append(
                {
                    "股票代码": position.raw_code,
                    "基础代码": position.code,
                    "股票简称": position.name,
                    "买入日期": position.entry_date.strftime("%Y%m%d"),
                    "卖出日期": trade_date.strftime("%Y%m%d"),
                    "买入原价": round(position.entry_price_raw, 4),
                    "买入执行价": round(position.entry_price_exec, 4),
                    "卖出执行价": round(exec_exit_price, 4),
                    "信号热度排名": position.signal_rank,
                    "分配资金": round(position.allocated_cash, 2),
                    "卖出回款": round(net_proceeds, 2),
                    "单笔净收益率": round(net_proceeds / position.allocated_cash - 1, 6),
                    "持有自然日": (trade_date - position.entry_date).days,
                }
            )
            positions.remove(position)

        position_value = 0.0
        for position in positions:
            history_df = histories.get(position.code, pd.DataFrame())
            quote, _ = module.get_quote_on_or_before(history_df, trade_date)
            position_value += module.estimate_position_value(position, quote)

        equity = cash + position_value
        equity_records.append(
            {
                "日期": trade_date,
                "现金": round(cash, 2),
                "持仓市值": round(position_value, 2),
                "总权益": round(equity, 2),
                "持仓数量": len(positions),
            }
        )

    equity_df = pd.DataFrame(equity_records)
    equity_df["净值"] = equity_df["总权益"] / module.INITIAL_CAPITAL
    equity_df["历史高点"] = equity_df["净值"].cummax()
    equity_df["回撤"] = equity_df["净值"] / equity_df["历史高点"] - 1
    trade_df = pd.DataFrame(trade_records)
    summary = {
        "初始资金": module.INITIAL_CAPITAL,
        "期末权益": float(equity_df.iloc[-1]["总权益"]),
        "期末净值": float(equity_df.iloc[-1]["净值"]),
        "总收益率": float(equity_df.iloc[-1]["净值"] - 1),
        "最大回撤": float(equity_df["回撤"].min()),
        "交易笔数": int(len(trade_df)),
        "胜率": float((trade_df["单笔净收益率"] > 0).mean()) if not trade_df.empty else math.nan,
        "平均单笔净收益率": float(trade_df["单笔净收益率"].mean()) if not trade_df.empty else math.nan,
        "收益回撤比": float((equity_df.iloc[-1]["净值"] - 1) / abs(equity_df["回撤"].min())) if equity_df["回撤"].min() < 0 else math.nan,
        "未平仓数量": len(positions),
    }
    return trade_df, equity_df, pd.DataFrame([summary])


def main() -> None:
    module = load_backtest_module()
    signal_df = module.load_factor_signal_data(FACTOR_CSV)
    market_df = pd.read_csv(MARKET_BREADTH_CSV, encoding="utf-8-sig")
    market_df["日期"] = pd.to_datetime(market_df["日期"])
    market_map = {
        pd.Timestamp(row["日期"]): row.to_dict()
        for _, row in market_df.iterrows()
    }
    candidate_book = module.build_candidate_book_with_config(
        signal_df=signal_df,
        breadth_df=pd.DataFrame(),
        buy_filters=module.NEW_FACTOR_CONFIG.buy_filters,
        named_conditions=module.NEW_FACTOR_CONFIG.named_conditions,
        market_filter=module.NEW_FACTOR_CONFIG.market_filter,
        max_positions=module.MAX_POSITIONS,
    )
    codes = sorted(candidate_book["基础代码"].dropna().astype(str).unique().tolist())
    histories = module.fetch_histories(codes)

    modes = {
        "基线": "baseline",
        "连跌两天停一天": "two_down_pause",
        "连跌两天半仓": "two_down_half",
        "回撤阈值控仓": "drawdown_control",
        "回撤更敏感控仓": "drawdown_tight",
        "市场宽度>=0才开仓": "breadth_ge_0",
        "市场宽度>=300才开仓": "breadth_ge_300",
        "候选均强度>=8.5才开仓": "quality_strength_ge_8_5",
        "候选至少2只且均强度>=8.5": "quality_count_ge_2_strength_ge_8_5",
    }

    rows = []
    for label, mode in modes.items():
        trade_df, equity_df, summary_df = run_backtest_with_risk(module, candidate_book, histories, mode, market_map)
        row = summary_df.copy()
        row.insert(0, "风控策略", label)
        rows.append(row)

        file_label = safe_name(label)
        export_equity = equity_df.copy()
        export_equity["日期"] = export_equity["日期"].dt.strftime("%Y%m%d")
        export_equity.to_csv(f"{OUTPUT_STEM}-{file_label}-净值.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(f"{OUTPUT_STEM}-{file_label}-交易明细.csv", index=False, encoding="utf-8-sig")

    result_df = pd.concat(rows, ignore_index=True)
    result_df = result_df.sort_values(["收益回撤比", "总收益率"], ascending=[False, False], na_position="last")
    result_df.to_csv(f"{OUTPUT_STEM}-结果.csv", index=False, encoding="utf-8-sig")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
