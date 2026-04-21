from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests


SCRIPT_PATH = Path(__file__).resolve().parent / "竞价爬升区间.py"
BASE_URL = "https://www.iwencai.com"


def load_cookie(script_path: Path | None = None) -> str:
    source_path = script_path or SCRIPT_PATH
    text = source_path.read_text(encoding="utf-8")
    match = re.search(r"cookie='([^']+)'", text)
    if not match:
        raise RuntimeError("未找到 cookie")
    return match.group(1)


def build_headers(cookie: str) -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "cookie": cookie,
        "Referer": "https://www.iwencai.com/",
        "Origin": "https://www.iwencai.com",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
    }


def post_question(question: str, cookie: str, perpage: int = 30) -> dict[str, Any]:
    payload = {
        "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
        "perpage": str(perpage),
        "page": 1,
        "source": "Ths_iwencai_Xuangu",
        "log_info": '{"input_type":"click"}',
        "version": "2.0",
        "secondary_intent": "stock",
        "question": question,
    }
    response = requests.post(
        urljoin(BASE_URL, "/customized/chart/get-robot-data"),
        json=payload,
        headers=build_headers(cookie),
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status_code") != 0:
        raise RuntimeError(f"问财查询失败: {result.get('status_msg')}")
    return result


def extract_footer_url(robot_data: dict[str, Any]) -> str:
    answer = robot_data.get("data", {}).get("answer", [])
    if not answer:
        raise RuntimeError("问财未返回 answer")
    txt_list = answer[0].get("txt", [])
    if not txt_list:
        raise RuntimeError("问财未返回 txt")
    content = txt_list[0].get("content")
    if not isinstance(content, dict):
        raise RuntimeError("问财 content 结构异常")
    components = content.get("components", [])
    if not components:
        raise RuntimeError("问财未返回 components")
    footer_url = components[0].get("config", {}).get("other_info", {}).get("footer_info", {}).get("url")
    if not footer_url:
        raise RuntimeError("问财未返回 footer url")
    return footer_url


def fetch_query_dataframe(question: str, cookie: str) -> pd.DataFrame:
    robot_data = post_question(question, cookie)
    footer_url = extract_footer_url(robot_data)
    response = requests.get(urljoin(BASE_URL, footer_url), headers=build_headers(cookie), timeout=30)
    response.raise_for_status()
    result = response.json()
    data = result.get("data") or result

    if isinstance(data, dict):
        for key_path in [
            ("answer",),
            ("datas",),
            ("list",),
            ("result", "data"),
            ("data", "datas"),
        ]:
            current: Any = data
            found = True
            for key in key_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    found = False
                    break
            if found and isinstance(current, list):
                return pd.DataFrame(current)

    raise RuntimeError(f"未识别的数据结构: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")


def merge_query_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    merged = valid_frames[0].copy()
    for frame in valid_frames[1:]:
        merge_key = None
        for candidate in ["股票代码", "code", "证券代码"]:
            if candidate in merged.columns and candidate in frame.columns:
                merge_key = candidate
                break
        if merge_key is None:
            raise RuntimeError("缺少可用于合并的股票代码列")
        merged = pd.merge(merged, frame, on=merge_key, how="inner", suffixes=("", "_drop"))
        merged = merged.filter(regex=r"^(?!.*_drop$)")
    return merged
