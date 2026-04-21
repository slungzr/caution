from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests


UNIFIED_WAP_URL = "https://www.iwencai.com/unifiedwap/unified-wap/v2/result/get-robot-data"
DEFAULT_SLEEP = 2.5
DEFAULT_RETRY = 3
DEFAULT_PERPAGE = 100

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.iwencai.com",
    "Referer": "https://www.iwencai.com/unifiedwap/result?w=%E4%B8%8A%E4%B8%80%E4%BA%A4%E6%98%93%E6%97%A5%E7%AB%9E%E4%BB%B7%E5%BC%BA%E5%BA%A6&querytype=stock&sign=1776613002120",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "hexin-v": "A2HMaNWfrNWCtQCiOEB1oAPTcCZ-Dtc5_4B5F8M3X8PGA49YC17l0I_SialQ",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Cookie": "other_uid=Ths_iwencai_Xuangu_7upib8nil3lgv9oqx1kb9k2g6mgo5dp3; cid=98ceb7f0b60eed590fd32d13d6b1d31e1776614455; ttype=WEB; user=MDpteF8xNzgyNTA0NDM6Ok5vbmU6NTAwOjE4ODI1MDQ0Mzo1LDEsNDA7NiwxLDQwOzcsMTExMTExMTExMTEwLDQwOzgsMTExMTAxMTEwMDAwMTExMTEwMDEwMDAwMDEwMDAwMDAsODk7MzMsMDAwMTAwMDAwMDAwLDg5OzM2LDEwMDExMTExMDAwMDExMDAxMDExMTExMSw4OTs0NiwwMDAwMTExMTEwMDAwMDExMTExMTExMTEsODk7NTEsMTEwMDAwMDAwMDAwMDAwMCw4OTs1OCwwMDAwMDAwMDAwMDAwMDAwMSw4OTs3OCwxLDg5Ozg3LDAwMDAwMDAwMDAwMDAwMDAwMDAxMDAwMCw4OTsxMTksMDAwMDAwMDAwMDAwMDAwMDAwMTAxMDAwMDAwMDAwMDAwMDAwMDAwMDAsODk7MTI1LDExLDg5OzEzMCwxMDEwMDAwMDAwMDAwLDg5OzQ0LDExLDQwOjE2Ojo6MTc4MjUwNDQzOjE3NzY2MTU4MjU6OjoxMzg1Nzc5ODYwOjYwNDgwMDowOjE4MWMxZjEyZjZiMWY5MTMyZTI1Y2QzMWRlMDAzZmFiNjpkZWZhdWx0XzU6MQ%3D%3D; userid=178250443; u_name=mx_178250443; escapename=mx_178250443; ticket=25753d75b907ca015075e2341da60a15; user_status=0; utk=b06b15e7c591b2bc4421bb6a0fdde37e; sess_tk=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6InNlc3NfdGtfMSIsImJ0eSI6InNlc3NfdGsifQ.eyJqdGkiOiJiNmZhMDNlMDFkZDM1Y2UyMzI5MTFmNmIyZmYxYzE4MTEiLCJpYXQiOjE3NzY2MTU4MjUsImV4cCI6MTc3NzIyMDYyNSwic3ViIjoiMTc4MjUwNDQzIiwiaXNzIjoidXBhc3MuaXdlbmNhaS5jb20iLCJhdWQiOiIyMDIwMTExODUyODg5MDcyIiwiYWN0Ijoib2ZjIiwiY3VocyI6IjAwOGU5YmU4MmE3OGJhY2UzZDc3MmU0ZDZlZGY0M2RhYjA0MDRlYzZiZWRiMzgxNzNlMDI3MjBkZjk0NmVjNGQifQ.1HVi-uFjytdjeP05K7eSZ5RTW7EjJqLdcLhV0cRiKWxlJ51Cw7gBY4nIMvxUxiEitTmnrKPfxK5VA-xlQW7Kvg; cuc=myv4ghgefxao; THSSESSID=014c470358e6299bb3024865ca; _clck=gcuw5r%7C2%7Cg5d%7C0%7C0; _clsk=kx4t9ur0z1gi%7C1776682887599%7C2%7C1%7C; v=A2HMaNWfrNWCtQCiOEB1oAPTcCZ-Dtc5_4B5F8M3X8PGA49YC17l0I_SialQ",
}


class WencaiUnifiedWapError(RuntimeError):
    pass


def _extract_components(payload: dict[str, Any]) -> list[dict[str, Any]]:
    content = payload.get("data", {}).get("answer", [{}])[0].get("txt", [{}])[0].get("content")
    if isinstance(content, str):
        content = json.loads(content)
    if not isinstance(content, dict):
        raise WencaiUnifiedWapError("unified-wap 返回内容结构异常")
    components = content.get("components", [])
    if not components:
        raise WencaiUnifiedWapError("unified-wap 未返回 components")
    return components


def _extract_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    components = _extract_components(payload)
    datas = components[0].get("data", {}).get("datas")
    if not isinstance(datas, list):
        raise WencaiUnifiedWapError("unified-wap 未返回 datas")
    return pd.DataFrame.from_dict(datas)


def query_unified_wap(question: str, perpage: int = DEFAULT_PERPAGE, page: int = 1, retry: int = DEFAULT_RETRY) -> pd.DataFrame:
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
        "log_info": '{"input_type":"typewrite"}',
        "rsh": "178250443",
    }

    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            response = requests.post(UNIFIED_WAP_URL, headers=DEFAULT_HEADERS, data=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("status_code") != 0:
                raise WencaiUnifiedWapError(f"unified-wap 返回错误: {result.get('status_msg', '未知错误')}")
            return _extract_dataframe(result)
        except Exception as exc:
            last_error = exc
            if attempt < retry:
                time.sleep(DEFAULT_SLEEP * attempt)
                continue
            raise WencaiUnifiedWapError(f"unified-wap 查询失败: {exc}") from exc
    raise WencaiUnifiedWapError(f"unified-wap 查询失败: {last_error}")


def merge_frames(frames: list[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    merged = valid_frames[0].copy()
    for frame in valid_frames[1:]:
        key = "code" if "code" in frame.columns else "股票代码"
        merge_key = "code" if "code" in merged.columns else "股票代码"
        if key != merge_key:
            frame = frame.rename(columns={key: merge_key})
        merged = pd.merge(merged, frame, on=merge_key, how=how, suffixes=("", "_drop"))
        merged = merged.filter(regex=r"^(?!.*_drop$)")
    return merged
