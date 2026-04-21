from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd


DEFAULT_API_URL = "https://openapi.iwencai.com/v1/query2data"
DEFAULT_LIMIT = 100
DEFAULT_RETRY = 4


class WencaiOpenAPIError(RuntimeError):
    pass


def get_api_config() -> tuple[str, str]:
    api_url = os.environ.get("IWENCAI_BASE_URL", "https://openapi.iwencai.com").rstrip("/") + "/v1/query2data"
    api_key = os.environ.get("IWENCAI_API_KEY", "").strip()
    if not api_key:
        raise WencaiOpenAPIError("缺少 IWENCAI_API_KEY 环境变量")
    return api_url or DEFAULT_API_URL, api_key


def query_wencai(
    query: str,
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    is_cache: str = "1",
    expand_index: str = "true",
    retry: int = DEFAULT_RETRY,
    retry_sleep: float = 1.5,
) -> dict[str, Any]:
    api_url, api_key = get_api_config()
    payload = {
        "query": query,
        "page": str(page),
        "limit": str(limit),
        "is_cache": str(is_cache),
        "expand_index": str(expand_index),
    }
    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        request = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            status_code = result.get("status_code", 0)
            if status_code != 0:
                status_msg = result.get("status_msg", "未知错误")
                if attempt < retry:
                    time.sleep(retry_sleep * attempt)
                    continue
                raise WencaiOpenAPIError(f"问财 OpenAPI 返回错误: {status_msg}")
            return result
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            last_error = exc
            if exc.code in {401, 403, 429, 500, 502, 503, 504} and attempt < retry:
                time.sleep(retry_sleep * attempt)
                continue
            raise WencaiOpenAPIError(f"问财 OpenAPI 请求失败: HTTP Error {exc.code}: {exc.reason}") from exc
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retry:
                time.sleep(retry_sleep * attempt)
                continue
            raise WencaiOpenAPIError(f"问财 OpenAPI 请求失败: {exc}") from exc

    raise WencaiOpenAPIError(f"问财 OpenAPI 请求失败: {last_error}")


def query_wencai_all_pages(query: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    first_page = query_wencai(query=query, page=1, limit=limit)
    frames = [pd.DataFrame(first_page.get("datas", []))]
    code_count = int(first_page.get("code_count", 0) or 0)
    if code_count <= limit:
        return frames[0]

    total_pages = (code_count + limit - 1) // limit
    for page in range(2, total_pages + 1):
        page_result = query_wencai(query=query, page=page, limit=limit)
        frames.append(pd.DataFrame(page_result.get("datas", [])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def detect_code_column(df: pd.DataFrame) -> str:
    for candidate in ["股票代码", "code", "证券代码"]:
        if candidate in df.columns:
            return candidate
    raise WencaiOpenAPIError("返回结果缺少股票代码列")


def merge_wencai_frames(frames: list[pd.DataFrame], how: str = "inner") -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    merged = valid_frames[0].copy()
    merge_key = detect_code_column(merged)
    for frame in valid_frames[1:]:
        frame_key = detect_code_column(frame)
        if frame_key != merge_key:
            frame = frame.rename(columns={frame_key: merge_key})
        merged = pd.merge(merged, frame, on=merge_key, how=how, suffixes=("", "_drop"))
        merged = merged.filter(regex=r"^(?!.*_drop$)")
    return merged


def query_split_and_merge(queries: list[str], limit: int = DEFAULT_LIMIT, how: str = "inner") -> pd.DataFrame:
    frames = [query_wencai_all_pages(query, limit=limit) for query in queries]
    return merge_wencai_frames(frames, how=how)
