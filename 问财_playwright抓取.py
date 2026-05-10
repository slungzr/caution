#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open iwencai in a real Chrome browser with Playwright and export table data.

Examples:
    python 问财_playwright抓取.py "昨日涨停"
    python 问财_playwright抓取.py "https://www.iwencai.com/unifiedwap/result?w=..."
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
STRATEGY_SCRIPT = BASE_DIR / "生成今日操作清单.py"


def require_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "当前 Python 环境未安装 playwright。\n"
            "请先执行：\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chrome\n"
            "然后重新运行本脚本。"
        ) from exc
    return sync_playwright, PlaywrightTimeoutError


def build_url(value: str) -> str:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return (
        "https://www.iwencai.com/unifiedwap/result?"
        f"w={quote(value)}&querytype=stock&sign={int(time.time() * 1000)}"
    )


def safe_name(text: str) -> str:
    text = re.sub(r"^https?://", "", text.strip())
    text = re.sub(r"[\\/:*?\"<>|\s%&=]+", "_", text)
    return text.strip("_")[:48] or "wencai"


def strip_html(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"(?i)<br\s*/?>", " ", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for col in columns:
        base = strip_html(col) or "column"
        seen[base] = seen.get(base, 0) + 1
        result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return result


def cell_to_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def dataframe_from_stockpick_payload(payload: dict[str, Any]) -> pd.DataFrame | None:
    data_block = payload.get("data", {})
    if not isinstance(data_block, dict):
        return None
    result_block = data_block.get("result", {})
    if not isinstance(result_block, dict):
        return None
    rows = result_block.get("result")
    if not isinstance(rows, list) or not rows:
        return None

    titles = result_block.get("title") or []
    index_ids = result_block.get("indexID") or []
    width = len(rows[0]) if isinstance(rows[0], list) else 0
    if titles and len(titles) == width:
        columns = unique_columns(titles)
    elif index_ids and len(index_ids) == width:
        columns = unique_columns(index_ids)
    else:
        columns = [f"col_{i}" for i in range(width)]
    return pd.DataFrame(rows, columns=columns).apply(lambda col: col.map(cell_to_csv_value))


def walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def dataframe_from_unified_payload(payload: dict[str, Any]) -> pd.DataFrame | None:
    answer = payload.get("answer")
    if isinstance(answer, dict):
        for node in walk_json(answer):
            datas = node.get("datas")
            if isinstance(datas, list) and datas and isinstance(datas[0], dict):
                return pd.DataFrame(datas).apply(lambda col: col.map(cell_to_csv_value))

    for node in walk_json(payload):
        datas = node.get("datas")
        if isinstance(datas, list) and datas and isinstance(datas[0], dict):
            return pd.DataFrame(datas).apply(lambda col: col.map(cell_to_csv_value))
        rows = node.get("result")
        titles = node.get("title")
        if isinstance(rows, list) and rows and isinstance(rows[0], list) and isinstance(titles, list):
            if len(titles) == len(rows[0]):
                return pd.DataFrame(rows, columns=unique_columns(titles)).apply(lambda col: col.map(cell_to_csv_value))
    return None


def payload_row_count(payload: dict[str, Any]) -> int:
    df = dataframe_from_payload(payload)
    return 0 if df is None else len(df)


def dataframe_from_payload(payload: Any) -> pd.DataFrame | None:
    if not isinstance(payload, dict):
        return None
    return dataframe_from_stockpick_payload(payload) or dataframe_from_unified_payload(payload)


def choose_best_payload(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    best_df: pd.DataFrame | None = None
    best_payload: dict[str, Any] | None = None
    for payload in payloads:
        df = dataframe_from_payload(payload)
        if df is None or df.empty:
            continue
        if best_df is None or len(df) > len(best_df):
            best_df = df
            best_payload = payload
    return best_df, best_payload


def combine_payloads(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    frames: list[pd.DataFrame] = []
    used_payload: dict[str, Any] | None = None
    for payload in payloads:
        df = dataframe_from_payload(payload)
        if df is None or df.empty:
            continue
        frames.append(df)
        used_payload = payload
    if not frames:
        return None, None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    key_columns = [col for col in ["股票代码", "股票简称"] if col in combined.columns]
    if key_columns:
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    else:
        combined = combined.drop_duplicates(keep="last")
    return combined.reset_index(drop=True), used_payload


def extract_dom_table(page) -> pd.DataFrame | None:
    rows = page.locator("table tr")
    count = rows.count()
    if count <= 0:
        return None

    data: list[list[str]] = []
    for i in range(count):
        cells = rows.nth(i).locator("th,td")
        row = [strip_html(cells.nth(j).inner_text(timeout=1000)) for j in range(cells.count())]
        if any(row):
            data.append(row)
    if not data:
        return None

    width = max(len(row) for row in data)
    data = [row + [""] * (width - len(row)) for row in data]
    header = data[0]
    body = data[1:] if len(data) > 1 else []
    if len(set(header)) >= max(2, width // 2):
        return pd.DataFrame(body, columns=unique_columns(header))
    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(width)])


def extract_visible_result(page, payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    df, used_payload = combine_payloads(payloads)
    if df is not None and not df.empty:
        return df, used_payload
    dom_df = extract_dom_table(page)
    return dom_df, None


def scroll_result_page(page) -> int:
    return page.evaluate(
        """
        () => {
          let touched = 0;
          window.scrollTo(0, document.body.scrollHeight);
          const nodes = Array.from(document.querySelectorAll('*'));
          for (const el of nodes) {
            const style = window.getComputedStyle(el);
            const scrollable = /(auto|scroll)/.test(style.overflowY || '') || /(auto|scroll)/.test(style.overflow || '');
            if (scrollable && el.scrollHeight > el.clientHeight + 20) {
              el.scrollTop = el.scrollHeight;
              touched += 1;
            }
          }
          return touched;
        }
        """
    )


def collect_next_pages(page, max_pages: int = 10) -> int:
    clicked = 0
    next_text = "\u4e0b\u9875"
    try:
        page.wait_for_selector(".pcwencai-pagination li", timeout=10000)
    except Exception:
        return clicked
    for _ in range(max(0, max_pages - 1)):
        next_items = page.locator(".pcwencai-pagination li").filter(has_text=next_text)
        if next_items.count() <= 0:
            break
        next_item = next_items.last
        classes = next_item.get_attribute("class") or ""
        if "disabled" in classes:
            break
        try:
            with page.expect_response(
                lambda response: "getDataList" in response.url and response.status == 200,
                timeout=15000,
            ):
                next_item.click()
            clicked += 1
            page.wait_for_timeout(1000)
        except Exception:
            break
    return clicked


def install_response_collector(
    page,
    payloads: list[dict[str, Any]],
    robot_requests: list[dict[str, Any]] | None = None,
) -> None:
    interesting = ("iwencai.com", "stockpick", "unified", "robot-data", "load-data", "getDataList")

    def on_response(response):
        url = response.url
        if not all(part in url for part in ("iwencai.com",)) or not any(part in url for part in interesting[1:]):
            return
        try:
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type and not any(
                k in url for k in ("load-data", "robot-data", "unified-wap", "getDataList")
            ):
                return
            payload = response.json()
        except Exception:
            return
        if isinstance(payload, dict):
            payloads.append(payload)
            if robot_requests is not None and "get-robot-data" in url:
                try:
                    robot_requests.append(
                        {
                            "url": url,
                            "post_data": response.request.post_data or "",
                            "headers": response.request.all_headers(),
                        }
                    )
                except Exception:
                    pass

    page.on("response", on_response)


def fetch_expanded_robot_payload(context, robot_requests: list[dict[str, Any]], perpage: int) -> dict[str, Any] | None:
    if not robot_requests:
        return None
    request_info = robot_requests[-1]
    post_data = request_info.get("post_data") or ""
    if not post_data:
        return None

    params = dict(parse_qsl(post_data, keep_blank_values=True))
    params["page"] = "1"
    params["perpage"] = str(perpage)
    if params.get("add_info"):
        try:
            add_info = json.loads(params["add_info"])
            add_info.setdefault("urp", {})
            add_info["urp"]["page"] = 1
            add_info["urp"]["perpage"] = perpage
            params["add_info"] = json.dumps(add_info, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            pass

    headers = {
        key: value
        for key, value in (request_info.get("headers") or {}).items()
        if key.lower() not in {"content-length", "host"}
    }
    headers["content-type"] = "application/x-www-form-urlencoded"
    response = context.request.post(
        request_info["url"],
        data=urlencode(params),
        headers=headers,
        timeout=30000,
    )
    if response.status != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


class WencaiPlaywrightClient:
    def __init__(
        self,
        profile_dir: str,
        timeout: int = 90,
        headless: bool = False,
        max_scrolls: int = 12,
        stable_scrolls: int = 3,
        max_pages: int = 10,
        expanded_perpage: int = 100,
    ):
        self.profile_dir = Path(profile_dir)
        if not self.profile_dir.is_absolute():
            self.profile_dir = BASE_DIR / self.profile_dir
        self.timeout = timeout
        self.headless = headless
        self.max_scrolls = max_scrolls
        self.stable_scrolls = stable_scrolls
        self.max_pages = max_pages
        self.expanded_perpage = expanded_perpage
        self._playwright = None
        self._context = None
        self._page = None
        self._timeout_error = None

    def __enter__(self):
        sync_playwright, PlaywrightTimeoutError = require_playwright()
        self._timeout_error = PlaywrightTimeoutError
        self._playwright = sync_playwright().start()
        launch_args = {
            "user_data_dir": str(self.profile_dir),
            "headless": self.headless,
            "viewport": {"width": 1440, "height": 900},
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            self._context = self._playwright.chromium.launch_persistent_context(channel="chrome", **launch_args)
        except Exception:
            self._context = self._playwright.chromium.launch_persistent_context(**launch_args)
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._context is not None:
            self._context.close()
        if self._playwright is not None:
            self._playwright.stop()

    def query(self, question_or_url: str) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
        if self._page is None or self._timeout_error is None:
            raise RuntimeError("Playwright client is not started")

        payloads: list[dict[str, Any]] = []
        robot_requests: list[dict[str, Any]] = []
        install_response_collector(self._page, payloads, robot_requests)
        url = build_url(question_or_url)
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)

        deadline = time.time() + self.timeout
        df: pd.DataFrame | None = None
        used_payload: dict[str, Any] | None = None
        pagination_done = False
        expanded_done = False
        last_rows = 0
        stable_count = 0
        scroll_count = 0
        while time.time() < deadline:
            df, used_payload = extract_visible_result(self._page, payloads)
            if df is not None and not df.empty:
                if not expanded_done and self._context is not None:
                    expanded_payload = fetch_expanded_robot_payload(
                        self._context,
                        robot_requests,
                        self.expanded_perpage,
                    )
                    if expanded_payload is not None:
                        payloads.append(expanded_payload)
                        df, used_payload = extract_visible_result(self._page, payloads)
                    expanded_done = True
                if not pagination_done:
                    collect_next_pages(self._page, self.max_pages)
                    pagination_done = True
                    df, used_payload = extract_visible_result(self._page, payloads)
                current_rows = len(df)
                if current_rows > last_rows:
                    last_rows = current_rows
                    stable_count = 0
                else:
                    stable_count += 1
                if scroll_count >= self.max_scrolls or stable_count >= self.stable_scrolls:
                    return df, used_payload, url

            scroll_result_page(self._page)
            scroll_count += 1
            try:
                self._page.wait_for_load_state("networkidle", timeout=3000)
            except self._timeout_error:
                pass
            time.sleep(1.2)

        df, used_payload = extract_visible_result(self._page, payloads)
        if df is None or df.empty:
            raise RuntimeError("未抓到问财数据。请确认浏览器页面能正常显示结果，且没有验证码/登录拦截。")
        return df, used_payload, url


def run_fetch(args: argparse.Namespace) -> int:
    sync_playwright, PlaywrightTimeoutError = require_playwright()
    url = build_url(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.profile_dir)
    if not profile_dir.is_absolute():
        profile_dir = BASE_DIR / profile_dir

    payloads: list[dict[str, Any]] = []
    robot_requests: list[dict[str, Any]] = []
    with sync_playwright() as p:
        launch_args = {
            "user_data_dir": str(profile_dir),
            "headless": args.headless,
            "viewport": {"width": 1440, "height": 900},
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            context = p.chromium.launch_persistent_context(channel="chrome", **launch_args)
        except Exception:
            context = p.chromium.launch_persistent_context(**launch_args)

        page = context.pages[0] if context.pages else context.new_page()
        install_response_collector(page, payloads, robot_requests)
        page.goto(url, wait_until="domcontentloaded", timeout=args.timeout * 1000)

        deadline = time.time() + args.timeout
        df: pd.DataFrame | None = None
        used_payload: dict[str, Any] | None = None
        pagination_done = False
        expanded_done = False
        last_rows = 0
        stable_count = 0
        scroll_count = 0
        while time.time() < deadline:
            df, used_payload = choose_best_payload(payloads)
            if df is not None and not df.empty:
                if not expanded_done:
                    expanded_payload = fetch_expanded_robot_payload(
                        context,
                        robot_requests,
                        args.expanded_perpage,
                    )
                    if expanded_payload is not None:
                        payloads.append(expanded_payload)
                    expanded_done = True
                if not pagination_done:
                    collect_next_pages(page, args.max_pages)
                    pagination_done = True
                df, used_payload = combine_payloads(payloads)
                current_rows = len(df)
                if current_rows > last_rows:
                    last_rows = current_rows
                    stable_count = 0
                else:
                    stable_count += 1
                if scroll_count >= args.max_scrolls or stable_count >= args.stable_scrolls:
                    break
            scroll_result_page(page)
            scroll_count += 1
            try:
                page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightTimeoutError:
                pass
            time.sleep(1.2)

        if df is None or df.empty:
            df = extract_dom_table(page)

        if args.keep_open:
            print("浏览器保持打开。处理完登录/验证后，可重新运行脚本抓取。")
            page.pause()

        context.close()

    if df is None or df.empty:
        raise RuntimeError("未抓到数据。请确认浏览器页面能正常显示结果，且没有验证码/登录拦截。")

    prefix = args.prefix or f"问财_playwright_{safe_name(args.input)}_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if used_payload is not None and not args.no_json:
        json_path.write_text(json.dumps(used_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"url: {url}")
    print(f"profile: {profile_dir}")
    print(f"rows_exported: {len(df)}")
    print(f"csv: {csv_path}")
    if used_payload is not None and not args.no_json:
        print(f"json: {json_path}")
    preview_cols = [c for c in ["股票代码", "股票简称", "现价(元)", "涨跌幅(%)"] if c in df.columns]
    if preview_cols:
        print(df[preview_cols].head(10).to_string(index=False))
    return 0


def load_strategy_module():
    if not STRATEGY_SCRIPT.exists():
        raise FileNotFoundError(f"未找到原策略脚本: {STRATEGY_SCRIPT}")
    spec = importlib.util.spec_from_file_location("strategy_original", STRATEGY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载原策略脚本: {STRATEGY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_strategy(args: argparse.Namespace) -> int:
    strategy = load_strategy_module()
    arg_min_auction_change = getattr(args, "min_auction_change", None)
    arg_max_auction_change = getattr(args, "max_auction_change", None)
    strategy_args = argparse.Namespace(
        trade_date=args.trade_date,
        push=args.push,
        no_push=args.no_push,
        top_n=args.top_n if args.top_n is not None else strategy.TOP_N,
        fixed_top_n=args.fixed_top_n,
        min_auction_ratio=(
            args.min_auction_ratio
            if args.min_auction_ratio is not None
            else strategy.MIN_AUCTION_TO_YESTERDAY_RATIO
        ),
        no_auction_ratio_filter=args.no_auction_ratio_filter,
        min_auction_change=(
            arg_min_auction_change
            if arg_min_auction_change is not None
            else getattr(strategy, "MIN_AUCTION_CHANGE", None)
        ),
        max_auction_change=(
            arg_max_auction_change
            if arg_max_auction_change is not None
            else getattr(strategy, "MAX_AUCTION_CHANGE", None)
        ),
        no_auction_change_filter=getattr(args, "no_auction_change_filter", False),
        min_unmatched_ratio=(
            args.min_unmatched_ratio
            if args.min_unmatched_ratio is not None
            else strategy.DEFAULT_MIN_UNMATCHED_RATIO
        ),
        industry_filter=(
            args.industry_filter
            if args.industry_filter is not None
            else strategy.DEFAULT_INDUSTRY_FILTER_ENABLED
        ),
        no_execution_advice=getattr(args, "no_execution_advice", False),
    )
    strategy.parse_args = lambda: strategy_args
    strategy.resolve_cookies = lambda: ["playwright"]

    with WencaiPlaywrightClient(
        args.profile_dir,
        timeout=args.timeout,
        headless=args.headless,
        max_scrolls=args.max_scrolls,
        stable_scrolls=args.stable_scrolls,
        max_pages=args.max_pages,
        expanded_perpage=args.expanded_perpage,
    ) as client:
        def query_wencai_playwright(question: str, cookies: list[str], pause_seconds: float = 1.5) -> pd.DataFrame:
            df, _payload, url = client.query(question)
            print(f"Playwright问财完成 rows={len(df)} url={url}")
            time.sleep(max(0.0, pause_seconds))
            return df

        strategy.query_wencai = query_wencai_playwright
        strategy.main()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Use Playwright/Chrome to fetch iwencai result table.")
    parser.add_argument("input", nargs="?", default="昨日涨停", help="问财 URL 或查询语句")
    parser.add_argument("--strategy", action="store_true", help="复用生成今日操作清单.py 的策略流程，问财数据改用 Playwright 抓取")
    parser.add_argument("--trade-date", default=None, help="传给生成今日操作清单.py 的交易日期 YYYYMMDD")
    parser.add_argument("--push", action="store_true", help="策略模式下允许历史回放推送")
    parser.add_argument("--no-push", action="store_true", help="策略模式下不发送 PushPlus")
    parser.add_argument("--top-n", type=int, default=None, help="策略模式下传给生成今日操作清单.py 的最大入选数")
    parser.add_argument("--fixed-top-n", action="store_true", help="策略模式下关闭动态持仓，严格使用 --top-n")
    parser.add_argument("--min-auction-ratio", type=float, default=None, help="策略模式下竞昨成交比最低阈值")
    parser.add_argument("--no-auction-ratio-filter", action="store_true", help="策略模式下关闭竞昨成交比过滤")
    parser.add_argument("--min-auction-change", type=float, default=None, help="策略模式下竞价涨幅最低阈值，默认跟随正式脚本")
    parser.add_argument("--max-auction-change", type=float, default=None, help="策略模式下竞价涨幅最高阈值，默认跟随正式脚本")
    parser.add_argument("--no-auction-change-filter", action="store_true", help="策略模式下关闭竞价涨幅过滤")
    parser.add_argument("--min-unmatched-ratio", type=float, default=None, help="策略模式下竞价未匹配占比最低阈值")
    parser.add_argument("--no-execution-advice", action="store_true", help="策略模式下不输出挂单建议列")
    industry_group = parser.add_mutually_exclusive_group()
    industry_group.add_argument(
        "--industry-filter",
        dest="industry_filter",
        action="store_true",
        default=None,
        help="策略模式下启用申万一级行业涨幅>0过滤",
    )
    industry_group.add_argument(
        "--no-industry-filter",
        dest="industry_filter",
        action="store_false",
        help="策略模式下关闭申万一级行业涨幅过滤",
    )
    parser.add_argument("--profile-dir", default=".playwright_wencai_profile", help="Chrome 持久化用户目录")
    parser.add_argument("--out-dir", default=str(BASE_DIR), help="输出目录")
    parser.add_argument("--prefix", default=None, help="输出文件名前缀")
    parser.add_argument("--timeout", type=int, default=90, help="等待数据秒数")
    parser.add_argument("--max-scrolls", type=int, default=12, help="每个问财页面最多向下滚动次数")
    parser.add_argument("--stable-scrolls", type=int, default=3, help="行数连续多少次不增长后停止")
    parser.add_argument("--max-pages", type=int, default=10, help="每个问财页面最多自动点击分页页数")
    parser.add_argument("--expanded-perpage", type=int, default=100, help="复用页面请求时尝试一次拉取的最大行数")
    parser.add_argument("--headless", action="store_true", help="无界面运行，不适合首次登录")
    parser.add_argument("--keep-open", action="store_true", help="抓取结束前暂停并保持浏览器打开")
    parser.add_argument("--no-json", action="store_true", help="只输出 CSV，不保存监听到的 JSON")
    args = parser.parse_args()
    if args.strategy:
        return run_strategy(args)
    return run_fetch(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
