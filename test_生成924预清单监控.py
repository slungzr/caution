from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("生成924预清单监控.py")
SPEC = importlib.util.spec_from_file_location("preopen_monitor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
preopen_monitor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preopen_monitor
SPEC.loader.exec_module(preopen_monitor)


class PreopenMonitorTests(unittest.TestCase):
    def test_preopen_queries_use_924_not_925_or_today_open(self) -> None:
        queries = preopen_monitor.build_preopen_queries(
            pd.Timestamp("2026-04-24"),
            pd.Timestamp("2026-04-23"),
            pd.Timestamp("2026-04-22"),
        )

        self.assertIn("4月24日9点24分最低价", queries["base"])
        self.assertNotIn("9点25分", queries["base"])
        self.assertNotIn("开盘价", queries["detail"])
        self.assertIn("4月24日竞价匹配价", queries["amount"])
        self.assertIn("4月23日成交金额", queries["amount"])

    def test_decorate_execution_columns_adds_limit_price_reference(self) -> None:
        selected = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "股票简称": ["测试股"],
                "竞价匹配价今日": [10.0],
            }
        )

        out = preopen_monitor.decorate_execution_columns(selected, 0.005)

        self.assertEqual(out.loc[0, "建议动作"], "9:24:50前确认后集合竞价买入")
        self.assertAlmostEqual(out.loc[0, "建议限价上限"], 10.05)
        self.assertIn("9:25最终清单变化", out.loc[0, "执行备注"])

    def test_decorate_execution_columns_observation_mode_does_not_set_limit(self) -> None:
        selected = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "股票简称": ["测试股"],
                "竞价匹配价今日": [10.0],
            }
        )

        out = preopen_monitor.decorate_execution_columns(selected, 0.005, False, "字段不完整")

        self.assertEqual(out.loc[0, "建议动作"], "仅预观察，等待9:25正式清单确认")
        self.assertTrue(pd.isna(out.loc[0, "建议限价上限"]))
        self.assertIn("字段不完整", out.loc[0, "执行备注"])

    def test_decorate_execution_columns_tolerates_missing_match_price(self) -> None:
        selected = pd.DataFrame({"股票代码": ["000001"], "股票简称": ["测试股"]})

        out = preopen_monitor.decorate_execution_columns(selected, 0.005)

        self.assertIn("建议限价上限", out.columns)
        self.assertTrue(pd.isna(out.loc[0, "建议限价上限"]))

    def test_compute_preopen_factors_tolerates_missing_auction_amount(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "成交金额昨日": [1_000_000_000],
                "成交量昨日": [2_000_000],
                "成交量前日": [1_000_000],
            }
        )

        out = preopen_monitor.compute_preopen_factors(df)

        self.assertIn("竞价匹配金额_openapi", out.columns)
        self.assertIn("竞昨成交比", out.columns)
        self.assertTrue(pd.isna(out.loc[0, "竞昨成交比"]))
        self.assertEqual(out.loc[0, "昨日前日成交量比"], 2.0)

    def test_assess_preopen_data_readiness_marks_observation_when_ratio_missing(self) -> None:
        df = pd.DataFrame({"竞价匹配金额_openapi": [pd.NA], "成交金额昨日": [1_000_000_000]})
        df = preopen_monitor.compute_preopen_factors(df)

        readiness = preopen_monitor.assess_preopen_data_readiness(df)

        self.assertEqual(readiness["预清单数据成熟度"], "观察")
        self.assertFalse(readiness["可集合竞价委托"])

    def test_assess_preopen_data_readiness_requires_unmatched_for_default_bid(self) -> None:
        df = pd.DataFrame({"竞价匹配金额_openapi": [60_000_000], "成交金额昨日": [1_000_000_000]})
        df = preopen_monitor.compute_preopen_factors(df)

        readiness = preopen_monitor.assess_preopen_data_readiness(df)

        self.assertEqual(readiness["预清单数据成熟度"], "半完整")
        self.assertTrue(readiness["竞昨成交比可用"])
        self.assertFalse(readiness["未匹配占比可用"])
        self.assertFalse(readiness["可集合竞价委托"])

    def test_assess_preopen_data_readiness_complete_allows_bid(self) -> None:
        df = pd.DataFrame(
            {
                "竞价匹配金额_openapi": [60_000_000],
                "竞价未匹配金额": [6_000_000],
                "成交金额昨日": [1_000_000_000],
            }
        )
        df = preopen_monitor.compute_preopen_factors(df)

        readiness = preopen_monitor.assess_preopen_data_readiness(df)

        self.assertEqual(readiness["预清单数据成熟度"], "完整")
        self.assertTrue(readiness["可集合竞价委托"])

    def test_observation_strategy_skips_amount_and_ratio_filters(self) -> None:
        df = pd.DataFrame(
            {
                "基础代码": ["000001", "000002"],
                "股票代码": ["000001", "000002"],
                "股票简称": ["一号", "二号"],
                "实体涨跌幅昨日": [1.0, 3.0],
                "实体涨跌幅前日": [2.0, 2.0],
                "个股热度排名昨日": [20, 10],
                "竞价涨幅今日": [1.0, 5.0],
                "竞价换手率今日": [0.5, 0.8],
            }
        )
        snapshot = {"日期": "2026-04-23", "市场20日高低差": 600, "开仓开关": "通过"}

        filtered, selected, status = preopen_monitor.apply_preopen_observation_strategy(
            df,
            snapshot,
            pd.Timestamp("2026-04-24"),
            "测试",
            top_n=3,
            industry_filter_enabled=False,
            min_unmatched_ratio=None,
            dynamic_top_n_enabled=True,
        )

        self.assertEqual(status["金额过滤后"], 2)
        self.assertEqual(status["竞昨过滤后"], 2)
        self.assertEqual(status["实体过滤后"], 1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(selected.iloc[0]["股票代码"], "000001")

    def test_export_preopen_outputs_writes_separate_files(self) -> None:
        selected = pd.DataFrame(
            {
                "排序名次": [1],
                "股票代码": ["000001"],
                "股票简称": ["测试股"],
                "竞价匹配价今日": [10.0],
                "建议限价上限": [10.05],
                "竞价匹配金额_openapi": [60_000_000],
                "竞价未匹配占比": [0.1],
                "竞昨成交比": [0.03],
                "个股热度排名昨日": [12],
                "建议动作": ["9:24:50前确认后集合竞价买入"],
                "建议权重": [1.0],
                "执行备注": ["测试"],
            }
        )
        status = {
            "交易日期": "2026-04-24",
            "采集时间": "2026-04-24T09:24:45",
            "采集轮次": 1,
            "市场快照日期": "2026-04-23",
            "市场20日高低差": 600,
            "开仓开关": "通过",
            "行业强度口径": "测试",
            "行业过滤启用": False,
            "竞昨成交比阈值": 0.022,
            "未匹配占比阈值": None,
            "动态持仓启用": True,
            "配置最大入选数": 3,
            "最大入选数": 3,
            "原始候选数": 1,
            "金额过滤后": 1,
            "竞昨过滤后": 1,
            "实体过滤后": 1,
            "行业过滤后": 1,
            "未匹配过滤后": 1,
            "最终候选数": 1,
            "入选数": 1,
            "结果说明": "测试",
            "建议限价上浮比例": "0.50%",
            "预清单数据成熟度": "完整",
            "可集合竞价委托": True,
            "竞价金额可用": True,
            "竞昨成交比可用": True,
            "未匹配占比可用": True,
            "数据成熟度说明": "测试",
        }
        queries = {"base": "base", "detail": "detail", "amount": "amount"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_base = preopen_monitor.BASE_DIR
            original_prefix = preopen_monitor.OUTPUT_PREFIX
            try:
                preopen_monitor.BASE_DIR = Path(tmp_dir)
                preopen_monitor.OUTPUT_PREFIX = Path(tmp_dir) / "924预清单"
                paths = preopen_monitor.export_preopen_outputs(
                    pd.Timestamp("2026-04-24"),
                    selected,
                    selected,
                    status,
                    queries,
                )
                self.assertTrue(paths["latest_csv"].exists())
                self.assertTrue(paths["dated_md"].exists())
                self.assertIn("这是 9:24 预清单", paths["dated_md"].read_text(encoding="utf-8"))
            finally:
                preopen_monitor.BASE_DIR = original_base
                preopen_monitor.OUTPUT_PREFIX = original_prefix


if __name__ == "__main__":
    unittest.main()
