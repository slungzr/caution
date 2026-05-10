from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("生成今日操作清单.py")
SPEC = importlib.util.spec_from_file_location("daily_operation_list", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
daily_operation_list = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = daily_operation_list
SPEC.loader.exec_module(daily_operation_list)


class DailyOperationListTests(unittest.TestCase):
    def test_amount_query_requests_yesterday_turnover_amount(self) -> None:
        queries = daily_operation_list.build_queries(
            pd.Timestamp("2026-04-24"),
            pd.Timestamp("2026-04-23"),
            pd.Timestamp("2026-04-22"),
        )

        self.assertIn("4月23日成交金额", queries["amount"])

    def test_compute_factors_uses_direct_yesterday_turnover_amount(self) -> None:
        df = pd.DataFrame(
            {
                "竞价匹配金额_openapi": [50_000_000],
                "竞价未匹配金额": [10_000_000],
                "成交金额昨日": [1_000_000_000],
                "成交量昨日": [1_000_000],
                "成交量前日": [2_000_000],
                "开盘价:不复权今日": [10],
                "竞价涨幅今日": [0],
            }
        )

        out = daily_operation_list.compute_factors(df)

        self.assertAlmostEqual(out.loc[0, "竞昨成交比"], 0.05)
        self.assertNotIn("昨收估算", out.columns)
        self.assertNotIn("昨日成交额估算", out.columns)
        self.assertNotIn("竞昨成交比估算", out.columns)

    def test_live_market_snapshot_refreshes_when_older_than_required_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "最新市场宽度.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "日期": "2026-04-24",
                        "市场20日高低差": -150,
                        "开仓开关": "不通过",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            original_snapshot_path = daily_operation_list.MARKET_SNAPSHOT_JSON
            original_refresh = getattr(daily_operation_list, "refresh_market_snapshot", None)
            refresh_calls: list[bool] = []

            def fake_refresh() -> None:
                refresh_calls.append(True)
                snapshot_path.write_text(
                    json.dumps(
                        {
                            "日期": "2026-04-27",
                            "市场20日高低差": 351,
                            "开仓开关": "通过",
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            daily_operation_list.MARKET_SNAPSHOT_JSON = snapshot_path
            daily_operation_list.refresh_market_snapshot = fake_refresh
            try:
                snapshot = daily_operation_list.read_market_snapshot(
                    pd.Timestamp("2026-04-28"),
                    False,
                    pd.Timestamp("2026-04-27"),
                )
            finally:
                daily_operation_list.MARKET_SNAPSHOT_JSON = original_snapshot_path
                if original_refresh is None:
                    delattr(daily_operation_list, "refresh_market_snapshot")
                else:
                    daily_operation_list.refresh_market_snapshot = original_refresh

            self.assertEqual(snapshot["日期"], "2026-04-27")
            self.assertEqual(snapshot["市场20日高低差"], 351)
            self.assertEqual(refresh_calls, [True])

    def test_refresh_market_snapshot_updates_snapshot_when_history_csv_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            snapshot_path = tmp_path / "最新市场宽度.json"
            history_path = tmp_path / "市场宽度历史.csv"
            script_path = tmp_path / "获取市场宽度.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "def fetch_market_breadth():",
                        "    return 'latest'",
                        "def read_history_if_exists(path):",
                        "    return 'existing'",
                        "def merge_history(existing_df, latest_df):",
                        "    return {'merged': True}",
                        "def export_history(history_df, csv_path):",
                        "    raise PermissionError(str(csv_path))",
                        "def export_snapshot(history_df, snapshot_path):",
                        "    snapshot = {'日期': '2026-04-27', '市场20日高低差': 351, '开仓开关': '通过'}",
                        "    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')",
                        "    return snapshot",
                    ]
                ),
                encoding="utf-8",
            )

            original_snapshot_path = daily_operation_list.MARKET_SNAPSHOT_JSON
            original_history_path = daily_operation_list.MARKET_HISTORY_CSV
            original_script_path = daily_operation_list.MARKET_WIDTH_SCRIPT
            daily_operation_list.MARKET_SNAPSHOT_JSON = snapshot_path
            daily_operation_list.MARKET_HISTORY_CSV = history_path
            daily_operation_list.MARKET_WIDTH_SCRIPT = script_path
            try:
                daily_operation_list.refresh_market_snapshot()
            finally:
                daily_operation_list.MARKET_SNAPSHOT_JSON = original_snapshot_path
                daily_operation_list.MARKET_HISTORY_CSV = original_history_path
                daily_operation_list.MARKET_WIDTH_SCRIPT = original_script_path

            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["日期"], "2026-04-27")
            self.assertEqual(snapshot["市场20日高低差"], 351)

    def test_push_content_uses_direct_auction_to_yesterday_turnover_ratio(self) -> None:
        selected_df = pd.DataFrame(
            {
                "排序名次": [1],
                "股票简称": ["测试股"],
                "股票代码": ["000001"],
                "建议权重": [1],
                "竞价匹配金额_openapi": [50_000_000],
                "竞价未匹配占比": [0.2],
                "竞昨成交比": [0.05],
                "竞价涨幅今日": 0,
                "个股热度排名昨日": [12],
            }
        )
        status = {
            "开仓开关": "通过",
            "市场20日高低差": 351,
            "行业强度口径": "测试",
            "原始候选数": 1,
            "金额过滤后": 1,
            "实体过滤后": 1,
            "行业过滤后": 1,
            "入选数": 1,
            "竞昨成交比阈值": 0.022,
            "配置最大入选数": 3,
            "最大入选数": 3,
            "动态持仓启用": True,
            "竞昨过滤后": 1,
            "行业过滤启用": False,
            "未匹配占比阈值": None,
            "未匹配过滤后": 1,
            "盘后弱承接风控阈值": -0.05,
            "结果说明": "测试",
        }

        content = daily_operation_list.build_pushplus_content(
            pd.Timestamp("2026-04-28"),
            selected_df,
            status,
        )

        self.assertIn("竞昨成交比: 0.0500", content)
        self.assertIn("竞昨成交比阈值: >=0.0220", content)
        self.assertIn("未匹配占比阈值: 关闭", content)
        self.assertIn("动态持仓: 启用 / 配置TOP3 -> 今日TOP3", content)
        self.assertIn("盘后弱承接风控", content)
        self.assertIn("<= -5.0%", content)
        self.assertIn("行业过滤: 关闭", content)
        self.assertIn("今日前3标的", content)

    def test_apply_strategy_filters_default_low_auction_ratio(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002", "000003"],
                "股票代码": ["000001", "000002", "000003"],
                "股票简称": ["低比", "达标", "高比"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000, 60_000_000],
                "竞昨成交比": [0.021, 0.022, 0.05],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [0.3, 0.2, 0.1],
                "实体涨跌幅昨日": [1.0, 1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0, 2.0],
                "申万一级行业涨跌幅": [1.0, 1.0, 1.0],
                "个股热度排名昨日": [10, 11, 12],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 600, "开仓开关": "通过"}

        filtered, selected, status = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
        )

        self.assertEqual(status["金额过滤后"], 3)
        self.assertEqual(status["竞昨成交比阈值"], 0.022)
        self.assertEqual(status["配置最大入选数"], 3)
        self.assertEqual(status["最大入选数"], 3)
        self.assertTrue(status["动态持仓启用"])
        self.assertFalse(status["行业过滤启用"])
        self.assertEqual(status["竞昨过滤后"], 2)
        self.assertIsNone(status["未匹配占比阈值"])
        self.assertEqual(status["未匹配过滤后"], 2)
        self.assertEqual(set(filtered["基础代码"]), {"000002", "000003"})
        self.assertEqual(len(selected), 2)
        self.assertIn("挂单建议", selected.columns)

    def test_apply_strategy_accepts_top_n_override(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002", "000003"],
                "股票代码": ["000001", "000002", "000003"],
                "股票简称": ["一号", "二号", "三号"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000, 60_000_000],
                "竞昨成交比": [0.03, 0.04, 0.05],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [0.3, 0.2, 0.1],
                "实体涨跌幅昨日": [1.0, 1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0, 2.0],
                "申万一级行业涨跌幅": [1.0, 1.0, 1.0],
                "个股热度排名昨日": [10, 11, 12],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 351, "开仓开关": "通过"}

        _filtered, selected, status = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
            top_n=2,
        )

        self.assertEqual(status["最大入选数"], 2)
        self.assertEqual(len(selected), 2)

    def test_apply_strategy_uses_dynamic_top_n_in_weaker_market(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002", "000003"],
                "股票代码": ["000001", "000002", "000003"],
                "股票简称": ["一号", "二号", "三号"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000, 60_000_000],
                "竞昨成交比": [0.03, 0.04, 0.05],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [0.3, 0.2, 0.1],
                "实体涨跌幅昨日": [1.0, 1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0, 2.0],
                "申万一级行业涨跌幅": [1.0, 1.0, 1.0],
                "个股热度排名昨日": [10, 11, 12],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 251, "开仓开关": "通过"}

        _filtered, selected_dynamic, status_dynamic = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
        )
        _filtered, selected_fixed, status_fixed = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
            dynamic_top_n_enabled=False,
        )

        self.assertEqual(status_dynamic["配置最大入选数"], 3)
        self.assertEqual(status_dynamic["最大入选数"], 2)
        self.assertEqual(len(selected_dynamic), 2)
        self.assertFalse(status_fixed["动态持仓启用"])
        self.assertEqual(status_fixed["最大入选数"], 3)
        self.assertEqual(len(selected_fixed), 3)

    def test_apply_strategy_uses_top1_in_low_positive_market(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002", "000003"],
                "股票代码": ["000001", "000002", "000003"],
                "股票简称": ["一号", "二号", "三号"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000, 60_000_000],
                "竞昨成交比": [0.03, 0.04, 0.05],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [0.3, 0.2, 0.1],
                "实体涨跌幅昨日": [1.0, 1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0, 2.0],
                "申万一级行业涨跌幅": [1.0, 1.0, 1.0],
                "个股热度排名昨日": [10, 11, 12],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 50, "开仓开关": "通过"}

        _filtered, selected, status = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
        )

        self.assertEqual(status["最大入选数"], 1)
        self.assertEqual(len(selected), 1)

    def test_execution_advice_marks_high_when_open_low_risk_is_high(self) -> None:
        row = pd.Series(
            {
                "排序名次": 2,
                "竞价涨幅今日": 0.8,
                "竞昨成交比": 0.025,
            }
        )

        advice = daily_operation_list.build_execution_advice(row, "强")

        self.assertEqual(advice["挂单建议"], "挂高")
        self.assertEqual(advice["建议挂单溢价"], daily_operation_list.EXECUTION_HIGH_PREMIUM)
        self.assertIn("竞价涨幅0~2%", advice["挂单建议理由"])

    def test_execution_advice_marks_low_for_middle_market_negative_auction(self) -> None:
        row = pd.Series(
            {
                "排序名次": 1,
                "竞价涨幅今日": -3.8,
                "竞昨成交比": 0.04,
            }
        )

        advice = daily_operation_list.build_execution_advice(row, "中")

        self.assertEqual(advice["挂单建议"], "挂低")
        self.assertEqual(advice["建议挂单溢价"], daily_operation_list.EXECUTION_LOW_DISCOUNT)
        self.assertEqual(advice["挂单上限溢价"], daily_operation_list.EXECUTION_LOW_DISCOUNT_MAX)
        self.assertIn("中市场负竞价", advice["挂单建议理由"])

    def test_execution_advice_marks_no_chase_when_auction_change_is_too_hot(self) -> None:
        row = pd.Series(
            {
                "排序名次": 1,
                "竞价涨幅今日": 5.8,
                "竞昨成交比": 0.026,
            }
        )

        advice = daily_operation_list.build_execution_advice(row, "强")

        self.assertEqual(advice["挂单建议"], "不追")
        self.assertEqual(advice["建议挂单溢价"], daily_operation_list.EXECUTION_NO_CHASE_PREMIUM)
        self.assertIn("竞价涨幅>5%", advice["挂单建议理由"])

    def test_apply_strategy_industry_filter_is_optional(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002"],
                "股票代码": ["000001", "000002"],
                "股票简称": ["弱行业", "强行业"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000],
                "竞昨成交比": [0.03, 0.04],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [0.3, 0.2],
                "实体涨跌幅昨日": [1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0],
                "申万一级行业涨跌幅": [-0.5, 1.0],
                "个股热度排名昨日": [10, 11],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 351, "开仓开关": "通过"}

        filtered_default, _selected_default, status_default = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
        )
        filtered_enabled, _selected_enabled, status_enabled = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
            industry_filter_enabled=True,
        )

        self.assertFalse(status_default["行业过滤启用"])
        self.assertEqual(set(filtered_default["基础代码"]), {"000001", "000002"})
        self.assertTrue(status_enabled["行业过滤启用"])
        self.assertEqual(set(filtered_enabled["基础代码"]), {"000002"})

    def test_apply_strategy_can_filter_min_unmatched_ratio(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002", "000003"],
                "股票代码": ["000001", "000002", "000003"],
                "股票简称": ["负未匹配", "低未匹配", "达标"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000, 60_000_000],
                "竞昨成交比": [0.03, 0.04, 0.05],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [-0.01, 0.003, 0.006],
                "实体涨跌幅昨日": [1.0, 1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0, 2.0],
                "申万一级行业涨跌幅": [1.0, 1.0, 1.0],
                "个股热度排名昨日": [10, 11, 12],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 351, "开仓开关": "通过"}

        filtered, selected, status = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
            min_unmatched_ratio=0.005,
        )

        self.assertEqual(status["未匹配占比阈值"], 0.005)
        self.assertEqual(status["未匹配过滤后"], 1)
        self.assertEqual(filtered["基础代码"].tolist(), ["000003"])
        self.assertEqual(selected["基础代码"].tolist(), ["000003"])

    def test_apply_strategy_can_disable_auction_ratio_filter(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002"],
                "股票代码": ["000001", "000002"],
                "股票简称": ["低比", "达标"],
                "竞价匹配金额_openapi": [60_000_000, 60_000_000],
                "竞昨成交比": [0.001, 0.02],
                "竞价涨幅今日": 0,
                "竞价未匹配占比": [0.3, 0.2],
                "实体涨跌幅昨日": [1.0, 1.0],
                "实体涨跌幅前日": [2.0, 2.0],
                "申万一级行业涨跌幅": [1.0, 1.0],
                "个股热度排名昨日": [10, 11],
            }
        )
        snapshot = {"日期": "2026-04-28", "市场20日高低差": 351, "开仓开关": "通过"}

        filtered, _selected, status = daily_operation_list.apply_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-28"),
            "测试行业口径",
            min_auction_ratio=None,
        )

        self.assertIsNone(status["竞昨成交比阈值"])
        self.assertEqual(status["竞昨过滤后"], 2)
        self.assertEqual(set(filtered["基础代码"]), {"000001", "000002"})


if __name__ == "__main__":
    unittest.main()
