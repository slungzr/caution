from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_JSON = BASE_DIR / "最新市场宽度.json"
STALE_WARNING_DAYS = 3


def load_snapshot(snapshot_path: Path) -> dict[str, object]:
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"未找到快照文件: {snapshot_path}。请先运行 获取市场宽度.py 更新市场宽度。"
        )

    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def main() -> None:
    snapshot = load_snapshot(SNAPSHOT_JSON)
    snapshot_date = pd.Timestamp(snapshot["日期"]).normalize()
    today = pd.Timestamp(datetime.now().date()).normalize()
    age_days = (today - snapshot_date).days
    stale_warning = age_days > STALE_WARNING_DAYS

    print(f"快照日期: {snapshot['日期']}")
    print(f"市场20日高低差: {snapshot['市场20日高低差']}")
    print(f"开仓开关: {snapshot['开仓开关']} (规则: {snapshot['规则']})")
    print(f"更新时间: {snapshot['更新时间']}")
    if stale_warning:
        print(f"状态提醒: 快照距今已 {age_days} 天，建议先运行 获取市场宽度.py 更新。")
    else:
        print(f"状态提醒: 快照距今 {age_days} 天，可直接使用。")


if __name__ == "__main__":
    main()
