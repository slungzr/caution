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
            "结果说明": "测试",
        }

        content = daily_operation_list.build_pushplus_content(
            pd.Timestamp("2026-04-28"),
            selected_df,
            status,
        )

        self.assertIn("竞昨成交比: 0.0500", content)


if __name__ == "__main__":
    unittest.main()
