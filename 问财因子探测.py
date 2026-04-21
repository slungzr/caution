from __future__ import annotations

import re
from pathlib import Path

import json
import pywencai
import requests

from wencai_direct import fetch_query_dataframe


SCRIPT_PATH = Path(__file__).resolve().parent / "竞价爬升区间.py"


def load_cookie() -> str:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    match = re.search(r"cookie='([^']+)'", text)
    if not match:
        raise RuntimeError("未找到 cookie")
    return match.group(1)


def direct_query(cookie: str, question: str) -> None:
    direct_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "cookie": cookie,
        "Referer": "https://www.iwencai.com/",
        "Origin": "https://www.iwencai.com",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
    }
    direct_data = {
        "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
        "perpage": "10",
        "page": 1,
        "source": "Ths_iwencai_Xuangu",
        "log_info": '{"input_type":"click"}',
        "version": "2.0",
        "secondary_intent": "stock",
        "question": question,
    }
    direct_response = requests.post(
        "https://www.iwencai.com/customized/chart/get-robot-data",
        json=direct_data,
        headers=direct_headers,
        timeout=30,
    )
    print(f"status={direct_response.status_code}")
    print(direct_response.text[:500])
    payload = direct_response.json()
    print("top_keys=", list(payload.keys()))
    answer = payload.get("data", {}).get("answer", [])
    print("answer_len=", len(answer))
    if answer:
        print("answer0_keys=", list(answer[0].keys()))
        txt = answer[0].get("txt", [])
        print("txt_len=", len(txt))
        if txt:
            print("txt0_keys=", list(txt[0].keys()))
            content = txt[0].get("content")
            print("content_type=", type(content).__name__)
            if isinstance(content, str):
                try:
                    content_json = json.loads(content)
                    print("content_json_keys=", list(content_json.keys()))
                except Exception as exc:
                    print(f"content_json_error={exc}")
            elif isinstance(content, dict):
                print("content_keys=", list(content.keys()))
                if "components" in content:
                    print("component_count=", len(content["components"]))
                    first_component = content["components"][0] if content["components"] else {}
                    print("first_component_keys=", list(first_component.keys()))
                    print("first_component_show_type=", first_component.get("show_type"))
                    footer_url = first_component.get("config", {}).get("other_info", {}).get("footer_info", {}).get("url")
                    print("first_component_footer_url=", footer_url)


def main() -> None:
    cookie = load_cookie()
    print("=== direct_simple ===")
    direct_query(cookie, "2025年9月10日竞价涨幅，2025年9月10日竞价换手率，2025年9月9日人气前100")
    print("=== direct_legacy ===")
    direct_query(
        cookie,
        "2025年9月10日9点25分最低价>2025年9月10日9点24分最高价，2025年9月10日9点24分最低价>=2025年9月10日9点23分最高价，2025年9月10日9点23分最低价>=2025年9月10日9点22分最高价，2025年9月10日9点22分最低价>=2025年9月10日9点21分最高价，2025年9月10日9点21分最低价>=2025年9月10日9点20分最高价,2025年9月10日竞价涨幅,2025年9月10日竞价换手率，2025年9月10日最低价，2025年9月9日人气前100",
    )
    print("=== direct_new_factors ===")
    direct_query(
        cookie,
        "2025年9月10日竞价强度，2025年9月10日竞价金额，2025年9月10日竞价换手率，2025年9月10日竞价量比，2025年9月10日竞价未匹配金额，2025年9月9日涨停，2025年9月9日炸板，2025年9月9日连续涨停天数，2025年9月9日个股热度排名前100",
    )
    print("=== direct_dataframe ===")
    df = fetch_query_dataframe(
        "2025年9月10日竞价强度，2025年9月10日竞价金额，2025年9月10日竞价换手率，2025年9月10日竞价量比，2025年9月10日竞价未匹配金额，2025年9月9日涨停，2025年9月9日炸板，2025年9月9日连续涨停天数，2025年9月9日个股热度排名前100",
        cookie,
    )
    print(f"dataframe_rows={len(df)}")
    print("dataframe_columns=")
    for column in df.columns:
        print(column)
    print(df.head(3).to_string())

    queries = {
        "legacy": (
            "2025年9月10日9点25分最低价>2025年9月10日9点24分最高价，"
            "2025年9月10日9点24分最低价>=2025年9月10日9点23分最高价，"
            "2025年9月10日9点23分最低价>=2025年9月10日9点22分最高价，"
            "2025年9月10日9点22分最低价>=2025年9月10日9点21分最高价，"
            "2025年9月10日9点21分最低价>=2025年9月10日9点20分最高价,"
            "2025年9月10日竞价涨幅,2025年9月10日竞价换手率，2025年9月10日最低价，2025年9月9日人气前100"
        ),
        "new_factors": (
            "2025年9月10日竞价强度，2025年9月10日竞价金额，2025年9月10日竞价换手率，"
            "2025年9月10日竞价量比，2025年9月10日竞价未匹配金额，"
            "2025年9月9日涨停，2025年9月9日炸板，2025年9月9日连续涨停天数，"
            "2025年9月9日个股热度排名前100"
        ),
    }

    for label, query in queries.items():
        print(f"=== {label} ===")
        result = pywencai.get(query=query, cookie=cookie)
        print(type(result).__name__)
        if result is None:
            print("result is None")
            continue
        if hasattr(result, "empty") and result.empty:
            print("result is empty")
            continue
        print(f"rows={len(result)}")
        print("columns=")
        for column in result.columns:
            print(column)
        print("preview=")
        print(result.head(3).to_string())


if __name__ == "__main__":
    main()
