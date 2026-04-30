#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use iwencai stockpick/load-data to fetch a query result and export CSV/JSON.

Examples:
    python 问财_stockpick_load_data.py "昨日涨停"
    python 问财_stockpick_load_data.py "昨日连板" --cookie-file wencai_cookie.txt
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
LOAD_DATA_URL = "https://www.iwencai.com/stockpick/load-data"


def parse_curl_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for match in re.finditer(r"-H\s+(['\"])(.*?)\1", text, flags=re.S):
        value = match.group(2).replace("\\\r\n", "").replace("\\\n", "")
        if ":" not in value:
            continue
        key, val = value.split(":", 1)
        headers[key.strip()] = val.strip()
    return headers


def clean_cookie_text(text: str) -> str:
    text = text.strip()
    headers = parse_curl_headers(text)
    if headers.get("Cookie"):
        return headers["Cookie"].strip()

    match = re.search(r"(?im)^\s*Cookie\s*:\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip().strip("'\"")

    return text.strip().strip("'\"")


def cookie_value(cookie: str, key: str) -> str:
    for part in cookie.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return ""


def read_python_string_var(path: Path, var_name: str) -> str:
    if not path.exists():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        tree = ast.parse(path.read_text(encoding="gbk", errors="ignore"))
    except Exception:
        return ""

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value.strip()
    return ""


def load_cookie(cookie_file: str | None = None) -> tuple[str, dict[str, str], str]:
    paths = []
    if cookie_file:
        paths.append(Path(cookie_file))
    paths.append(BASE_DIR / "wencai_cookie.txt")

    for path in paths:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cookie = clean_cookie_text(raw)
        if cookie:
            return cookie, parse_curl_headers(raw), str(path)

    env_cookie = os.environ.get("WENCAI_COOKIE", "").strip()
    if env_cookie:
        return clean_cookie_text(env_cookie), parse_curl_headers(env_cookie), "env:WENCAI_COOKIE"

    fallback_vars = [
        (BASE_DIR / "竞价因子补充_pywencai.py", "COOKIE"),
        (BASE_DIR / "wencai_unifiedwap.py", "Cookie"),
        (BASE_DIR / "竞价爬升区间.py", "cookie"),
    ]
    for path, var_name in fallback_vars:
        value = read_python_string_var(path, var_name)
        if value:
            return clean_cookie_text(value), parse_curl_headers(value), f"{path.name}:{var_name}"

    raise RuntimeError("未找到 Cookie：请把浏览器复制的 Cookie 或整段 curl 保存到 wencai_cookie.txt")


def build_headers(cookie: str, raw_headers: dict[str, str], question: str) -> dict[str, str]:
    encoded = quote(question)
    referer = (
        "https://www.iwencai.com/stockpick/search?"
        f"rsh=3&typed=1&preParams=&ts=1&f=1&qs=result_rewrite&selfsectsn="
        f"&querytype=stock&searchfilter=&tid=stockpick&w={encoded}"
    )
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Referer": referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }
    headers.update(raw_headers)
    headers["Cookie"] = cookie
    headers["Referer"] = raw_headers.get("Referer", referer)
    hexin_v = raw_headers.get("hexin-v") or cookie_value(cookie, "v")
    if hexin_v:
        headers["hexin-v"] = hexin_v
    return headers


def load_stockpick_data(question: str, cookie: str, raw_headers: dict[str, str]) -> dict[str, Any]:
    params = {
        "rsh": "3",
        "typed": "0",
        "preParams": "",
        "ts": "1",
        "f": "1",
        "qs": "result_original",
        "selfsectsn": "",
        "querytype": "stock",
        "searchfilter": "",
        "tid": "stockpick",
        "w": question,
        "queryarea": "",
    }
    headers = build_headers(cookie, raw_headers, question)
    response = requests.get(LOAD_DATA_URL, params=params, headers=headers, timeout=30)
    if response.status_code != 200:
        preview = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"load-data 请求失败: HTTP {response.status_code} {preview}")

    try:
        payload = response.json()
    except ValueError as exc:
        preview = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"load-data 返回不是 JSON: {preview}") from exc

    if not payload.get("success"):
        raise RuntimeError(f"load-data success=false: {json.dumps(payload, ensure_ascii=False)[:500]}")
    return payload


def strip_html(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"(?i)<br\s*/?>", " ", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for col in columns:
        base = strip_html(col) or "column"
        seen[base] = seen.get(base, 0) + 1
        result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return result


def cell_to_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    result_block = payload.get("data", {}).get("result", {})
    rows = result_block.get("result") or []
    titles = result_block.get("title") or []
    index_ids = result_block.get("indexID") or []

    if titles and rows and len(titles) == len(rows[0]):
        columns = unique_columns(titles)
    elif index_ids and rows and len(index_ids) == len(rows[0]):
        columns = unique_columns(index_ids)
    else:
        width = max((len(row) for row in rows), default=0)
        columns = [f"col_{i}" for i in range(width)]

    df = pd.DataFrame(rows, columns=columns)
    return df.apply(lambda col: col.map(cell_to_csv_value))


def safe_name(text: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|\s]+", "_", text.strip())
    return value.strip("_")[:40] or "query"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch iwencai stockpick/load-data result.")
    parser.add_argument("query", nargs="?", default="昨日涨停", help="问财查询语句，默认：昨日涨停")
    parser.add_argument("--cookie-file", default=None, help="Cookie 文件，默认读取 wencai_cookie.txt")
    parser.add_argument("--out-dir", default=str(BASE_DIR), help="输出目录")
    parser.add_argument("--prefix", default=None, help="输出文件名前缀")
    parser.add_argument("--no-json", action="store_true", help="只输出 CSV，不保存完整 JSON")
    args = parser.parse_args()

    cookie, raw_headers, source = load_cookie(args.cookie_file)
    payload = load_stockpick_data(args.query, cookie, raw_headers)
    df = payload_to_dataframe(payload)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"问财_stockpick_{safe_name(args.query)}_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if not args.no_json:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result_block = payload.get("data", {}).get("result", {})
    print(f"query: {args.query}")
    print(f"cookie_source: {source}")
    print(f"total: {result_block.get('total', len(df))}")
    print(f"rows_exported: {len(df)}")
    print(f"csv: {csv_path}")
    if not args.no_json:
        print(f"json: {json_path}")
    if not df.empty:
        preview_cols = [c for c in ["股票代码", "股票简称", "现价(元)", "涨跌幅(%)"] if c in df.columns]
        if preview_cols:
            print(df[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
