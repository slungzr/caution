from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from urllib.parse import quote
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = BASE_DIR / "生成今日操作清单.py"
UNIFIED_WAP_URL = "https://www.iwencai.com/unifiedwap/unified-wap/v2/result/get-robot-data"


def load_original_module():
    spec = importlib.util.spec_from_file_location("daily_operation_list_original", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载原脚本: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_cookie_text(text: str) -> str:
    value = text.strip()
    match = re.search(r"""(?is)(?:^|\s)-H\s+(['"])Cookie:\s*(.*?)\1""", value)
    if match:
        return match.group(2).strip()
    match = re.search(r"""(?is)^Cookie:\s*(.*)$""", value)
    if match:
        return match.group(1).strip()
    return value


def parse_curl_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for match in re.finditer(r"""(?is)(?:^|\s)-H\s+(['"])(.*?)\1""", text):
        header = match.group(2).strip()
        name, sep, value = header.partition(":")
        if sep:
            headers[name.strip()] = value.strip()
    return headers


def cookie_value(cookie: str, key: str) -> str | None:
    for item in clean_cookie_text(cookie).split(";"):
        name, sep, value = item.strip().partition("=")
        if sep and name == key:
            return value
    return None


def build_unified_headers(cookie: str, question: str | None = None) -> dict[str, str]:
    cleaned_cookie = clean_cookie_text(cookie)
    curl_headers = parse_curl_headers(cookie)
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.iwencai.com",
        "Referer": "https://www.iwencai.com/unifiedwap/result?querytype=stock",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Cookie": cleaned_cookie,
    }
    headers.update(curl_headers)
    headers["Cookie"] = clean_cookie_text(headers.get("Cookie", cleaned_cookie))
    hexin_v = cookie_value(cleaned_cookie, "v")
    if hexin_v:
        headers["hexin-v"] = hexin_v
    if question:
        headers["Referer"] = f"https://www.iwencai.com/unifiedwap/result?w={quote(question)}&querytype=stock"
    headers.pop("Content-Length", None)
    headers.pop("Host", None)
    return headers


def extract_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    content = payload.get("data", {}).get("answer", [{}])[0].get("txt", [{}])[0].get("content")
    if isinstance(content, str):
        content = json.loads(content)
    if not isinstance(content, dict):
        return pd.DataFrame()
    components = content.get("components", [])
    if not components:
        return pd.DataFrame()
    datas = components[0].get("data", {}).get("datas")
    if not isinstance(datas, list):
        return pd.DataFrame()
    return pd.DataFrame.from_dict(datas)


def query_unified_wap(question: str, cookie: str, perpage: int = 100, page: int = 1) -> pd.DataFrame:
    cleaned_cookie = clean_cookie_text(cookie)
    payload = {
        "source": "Ths_iwencai_Xuangu",
        "version": "2.0",
        "query_area": "",
        "block_list": "",
        "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
        "question": question,
        "perpage": str(perpage),
        "page": str(page),
        "secondary_intent": "stock",
        "log_info": '{"input_type":"click"}',
        "rsh": cookie_value(cleaned_cookie, "userid") or "",
    }
    response = requests.post(
        UNIFIED_WAP_URL,
        headers=build_unified_headers(cookie, question),
        data=payload,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status_code") != 0:
        raise RuntimeError(f"unifiedwap 返回错误: {result.get('status_msg')}")
    return extract_dataframe(result)


def install_unifiedwap_fallback(module) -> None:
    def query_wencai(question: str, cookies: list[str], pause_seconds: float = 1.5) -> pd.DataFrame:
        errors: list[str] = []
        for index, raw_cookie in enumerate(cookies, start=1):
            cookie = clean_cookie_text(raw_cookie)
            try:
                df = module.fetch_query_dataframe(question, cookie)
                if df is not None and not df.empty:
                    time.sleep(max(0.0, pause_seconds))
                    return df
                errors.append(f"cookie{index}:direct_empty")
            except Exception as exc:
                errors.append(f"cookie{index}:direct={exc}")

            try:
                df = module.pywencai.get(query=question, cookie=cookie)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    time.sleep(max(0.0, pause_seconds))
                    return df
                errors.append(f"cookie{index}:pywencai_empty")
            except Exception as exc:
                errors.append(f"cookie{index}:pywencai={exc}")

            try:
                df = query_unified_wap(question, cookie)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    time.sleep(max(0.0, pause_seconds))
                    return df
                errors.append(f"cookie{index}:unifiedwap_empty")
            except Exception as exc:
                errors.append(f"cookie{index}:unifiedwap={exc}")

        raise RuntimeError(f"问财查询失败: {'; '.join(errors)}")

    module.query_wencai = query_wencai


def main() -> None:
    module = load_original_module()
    install_unifiedwap_fallback(module)
    module.main()


if __name__ == "__main__":
    main()
