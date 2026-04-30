from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("竞价市值影响探索.py")
SPEC = importlib.util.spec_from_file_location("market_cap_explorer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
market_cap_explorer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = market_cap_explorer
SPEC.loader.exec_module(market_cap_explorer)


class MarketCapExplorerTests(unittest.TestCase):
    def test_normalize_amount_series_converts_chinese_units_to_yuan(self) -> None:
        values = pd.Series(["91.20亿", "3,500.5万", "120000000", "-", None])

        out = market_cap_explorer.normalize_amount_series(values)

        self.assertEqual(out.iloc[0], 9_120_000_000)
        self.assertEqual(out.iloc[1], 35_005_000)
        self.assertEqual(out.iloc[2], 120_000_000)
        self.assertTrue(pd.isna(out.iloc[3]))
        self.assertTrue(pd.isna(out.iloc[4]))

    def test_market_cap_bins_build_expected_filter_ranges(self) -> None:
        bins = market_cap_explorer.build_fixed_market_cap_bins("昨日a股流通市值")

        self.assertEqual(bins[0].name, "全样本")
        self.assertEqual(bins[1].name, "<20亿")
        self.assertEqual(bins[1].filters, {"昨日a股流通市值": {"max": 2_000_000_000}})
        self.assertEqual(
            bins[3].filters,
            {"昨日a股流通市值": {"min": 5_000_000_000, "max": 10_000_000_000}},
        )

    def test_find_market_cap_column_prefers_existing_yesterday_float_market_cap(self) -> None:
        df = pd.DataFrame(
            {
                "成交额昨日": [1],
                "a股市值(不含限售股)昨日": [2],
                "总市值昨日": [3],
            }
        )

        self.assertEqual(
            market_cap_explorer.find_market_cap_column(df),
            "a股市值(不含限售股)昨日",
        )

    def test_threshold_bins_build_lower_bound_scans(self) -> None:
        bins = market_cap_explorer.build_threshold_market_cap_bins("昨日a股流通市值", thresholds_yi=[20, 50])

        self.assertEqual([item.name for item in bins], ["全样本", ">=20亿", ">=50亿"])
        self.assertEqual(bins[1].filters, {"昨日a股流通市值": {"min": 2_000_000_000}})
        self.assertEqual(bins[2].filters, {"昨日a股流通市值": {"min": 5_000_000_000}})

    def test_rows_needing_market_cap_fetch_respects_missing_and_date_range(self) -> None:
        df = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
                "基础代码": ["000001", "000002", "000003"],
                "昨日a股流通市值": [pd.NA, pd.NA, 10],
            }
        )

        out = market_cap_explorer.rows_needing_market_cap_fetch(
            df,
            "昨日a股流通市值",
            start_date=pd.Timestamp("2026-04-02"),
            end_date=pd.Timestamp("2026-04-03"),
        )

        self.assertEqual(out["基础代码"].tolist(), ["000002"])

    def test_market_cap_queries_include_year(self) -> None:
        query = market_cap_explorer.market_cap_query(pd.Timestamp("2024-05-08"))

        self.assertIn("2024年5月8日a股流通市值", query)
        self.assertIn("2024年5月8日个股热度排名前100", query)

    def test_code_market_cap_query_uses_codes_and_year(self) -> None:
        query = market_cap_explorer.code_market_cap_query(["600678", "300719"], pd.Timestamp("2024-05-08"))

        self.assertEqual(query, "600678、300719，2024年5月8日a股流通市值")

    def test_standardize_market_cap_frame_accepts_code_column(self) -> None:
        df = pd.DataFrame(
            {
                "代码": ["600678"],
                "名称": ["测试"],
                "a股流通市值\r": ["91.2亿"],
            }
        )

        out = market_cap_explorer.standardize_market_cap_frame(
            df,
            pd.Timestamp("2024-05-09"),
            lambda value: str(value),
        )

        self.assertEqual(out.loc[0, "基础代码"], "600678")
        self.assertEqual(out.loc[0, "昨日a股流通市值"], 9_120_000_000)


if __name__ == "__main__":
    unittest.main()
