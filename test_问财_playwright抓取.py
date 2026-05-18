from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("问财_playwright抓取.py")
SPEC = importlib.util.spec_from_file_location("wencai_playwright_fetch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
wencai_playwright_fetch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wencai_playwright_fetch
SPEC.loader.exec_module(wencai_playwright_fetch)


class WencaiPlaywrightFetchTests(unittest.TestCase):
    def test_strategy_mode_passes_current_strategy_args(self) -> None:
        captured: dict[str, object] = {}

        class FakeStrategy:
            TOP_N = 2
            MIN_AUCTION_TO_YESTERDAY_RATIO = 0.022
            MIN_AUCTION_CHANGE = -9.0
            MAX_AUCTION_CHANGE = None
            MIN_MARKET_60_HIGH_LOW_DIFF = 50
            TRADE_RESULT_CSV = "实盘交易记录.csv"
            LOSS_COOLDOWN_ENABLED = True
            LOSS_COOLDOWN_CONSECUTIVE_LOSSES = 2
            LOSS_COOLDOWN_DAYS = 1
            DYNAMIC_TOP_N_ENABLED = False
            DYNAMIC_AUCTION_RATIO_ENABLED = True
            DEFAULT_MIN_UNMATCHED_RATIO = None
            DEFAULT_INDUSTRY_FILTER_ENABLED = False
            PREV_BODY_MIN = -5.0

            def parse_args(self):
                raise AssertionError("run_strategy should replace parse_args")

            def main(self) -> None:
                strategy_args = self.parse_args()
                captured.update(vars(strategy_args))

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def query(self, question):
                raise AssertionError("fake strategy main should not query")

        original_load_strategy = wencai_playwright_fetch.load_strategy_module
        original_client = wencai_playwright_fetch.WencaiPlaywrightClient
        wencai_playwright_fetch.load_strategy_module = lambda: FakeStrategy()
        wencai_playwright_fetch.WencaiPlaywrightClient = FakeClient
        try:
            result = wencai_playwright_fetch.run_strategy(
                Namespace(
                    trade_date="20260424",
                    push=True,
                    no_push=False,
                    top_n=2,
                    fixed_top_n=True,
                    min_auction_ratio=0.03,
                    no_auction_ratio_filter=False,
                    min_auction_change=-8.5,
                    max_auction_change=6.0,
                    no_auction_change_filter=False,
                    min_market60_high_low_diff=80,
                    no_market60_filter=False,
                    trade_result_file="custom_trades.csv",
                    loss_cooldown_consecutive_losses=3,
                    loss_cooldown_days=2,
                    no_loss_cooldown=True,
                    min_unmatched_ratio=0.005,
                    industry_filter=True,
                    profile_dir=".profile",
                    timeout=1,
                    headless=True,
                    max_scrolls=1,
                    stable_scrolls=1,
                    max_pages=1,
                    expanded_perpage=50,
                )
            )
        finally:
            wencai_playwright_fetch.load_strategy_module = original_load_strategy
            wencai_playwright_fetch.WencaiPlaywrightClient = original_client

        self.assertEqual(result, 0)
        self.assertEqual(captured["trade_date"], "20260424")
        self.assertEqual(captured["top_n"], 2)
        self.assertTrue(captured["fixed_top_n"])
        self.assertEqual(captured["min_auction_ratio"], 0.03)
        self.assertEqual(captured["min_auction_change"], -8.5)
        self.assertEqual(captured["max_auction_change"], 6.0)
        self.assertEqual(captured["min_market60_high_low_diff"], 80)
        self.assertFalse(captured["no_market60_filter"])
        self.assertEqual(captured["trade_result_file"], "custom_trades.csv")
        self.assertEqual(captured["loss_cooldown_consecutive_losses"], 3)
        self.assertEqual(captured["loss_cooldown_days"], 2)
        self.assertTrue(captured["no_loss_cooldown"])
        self.assertEqual(captured["min_unmatched_ratio"], 0.005)
        self.assertTrue(captured["industry_filter"])

    def test_strategy_mode_uses_strategy_defaults_when_not_overridden(self) -> None:
        captured: dict[str, object] = {}

        class FakeStrategy:
            TOP_N = 2
            MIN_AUCTION_TO_YESTERDAY_RATIO = 0.022
            MIN_AUCTION_CHANGE = -9.0
            MAX_AUCTION_CHANGE = None
            MIN_MARKET_60_HIGH_LOW_DIFF = 50
            TRADE_RESULT_CSV = "实盘交易记录.csv"
            LOSS_COOLDOWN_ENABLED = True
            LOSS_COOLDOWN_CONSECUTIVE_LOSSES = 2
            LOSS_COOLDOWN_DAYS = 1
            DYNAMIC_TOP_N_ENABLED = False
            DYNAMIC_AUCTION_RATIO_ENABLED = True
            DEFAULT_MIN_UNMATCHED_RATIO = None
            DEFAULT_INDUSTRY_FILTER_ENABLED = False
            PREV_BODY_MIN = -5.0

            def main(self) -> None:
                captured.update(vars(self.parse_args()))

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        original_load_strategy = wencai_playwright_fetch.load_strategy_module
        original_client = wencai_playwright_fetch.WencaiPlaywrightClient
        wencai_playwright_fetch.load_strategy_module = lambda: FakeStrategy()
        wencai_playwright_fetch.WencaiPlaywrightClient = FakeClient
        try:
            wencai_playwright_fetch.run_strategy(
                Namespace(
                    trade_date=None,
                    push=False,
                    no_push=True,
                    top_n=None,
                    fixed_top_n=None,
                    min_auction_ratio=None,
                    no_auction_ratio_filter=False,
                    min_auction_change=None,
                    max_auction_change=None,
                    no_auction_change_filter=False,
                    min_market60_high_low_diff=None,
                    no_market60_filter=False,
                    trade_result_file=None,
                    loss_cooldown_consecutive_losses=None,
                    loss_cooldown_days=None,
                    no_loss_cooldown=False,
                    min_unmatched_ratio=None,
                    industry_filter=None,
                    profile_dir=".profile",
                    timeout=1,
                    headless=True,
                    max_scrolls=1,
                    stable_scrolls=1,
                    max_pages=1,
                    expanded_perpage=50,
                )
            )
        finally:
            wencai_playwright_fetch.load_strategy_module = original_load_strategy
            wencai_playwright_fetch.WencaiPlaywrightClient = original_client

        self.assertEqual(captured["top_n"], 2)
        self.assertTrue(captured["fixed_top_n"])
        self.assertEqual(captured["min_auction_ratio"], 0.022)
        self.assertTrue(captured["dynamic_auction_ratio"])
        self.assertEqual(captured["min_auction_change"], -9.0)
        self.assertIsNone(captured["max_auction_change"])
        self.assertEqual(captured["min_market60_high_low_diff"], 50)
        self.assertFalse(captured["no_market60_filter"])
        self.assertEqual(captured["trade_result_file"], "实盘交易记录.csv")
        self.assertEqual(captured["loss_cooldown_consecutive_losses"], 2)
        self.assertEqual(captured["loss_cooldown_days"], 1)
        self.assertFalse(captured["no_loss_cooldown"])
        self.assertIsNone(captured["min_unmatched_ratio"])
        self.assertFalse(captured["industry_filter"])
        self.assertEqual(captured["prev_body_min"], -5.0)


if __name__ == "__main__":
    unittest.main()
