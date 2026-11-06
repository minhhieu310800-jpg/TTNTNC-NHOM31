"""Ghi và đọc log thi đấu dạng JSONL: mỗi ván một dòng.

Mỗi dòng gồm engine_version, rule, size, opening_id, seed, agent_x/agent_o
(kèm cấu hình), danh sách nước đi (action, time_ms, stats lấy từ last_stats)
và result. Thông tin máy và cấu hình của mỗi lần chạy ghi ở file .meta.jsonl.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Đổi giá trị (numpy, tuple, inf...) về dạng ghi được ra JSON."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


class JsonlLogger:
    """Ghi nối tiếp từng ván vào file JSONL, flush ngay để không mất dữ liệu khi bị ngắt."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(to_jsonable(record), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Đọc từng ván; bỏ qua dòng hỏng (ví dụ dòng cuối bị cắt khi tiến trình bị dừng)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_records(path))


def completed_game_ids(path: str | Path) -> set[str]:
    """Các game_id đã có trong log, dùng để chạy tiếp khi bị ngắt."""
    return {r["game_id"] for r in iter_records(path) if "game_id" in r}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _total_ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (AttributeError, ValueError, OSError):
        return None


def hardware_info() -> dict[str, Any]:
    """Cấu hình máy và phần mềm của lần chạy (đề yêu cầu khai báo phần cứng)."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def write_run_meta(log_path: str | Path, meta: dict[str, Any]) -> Path:
    """Ghi một dòng meta cho lần chạy vào <log>.meta.jsonl."""
    log_path = Path(log_path)
    meta_path = log_path.with_suffix(".meta.jsonl")
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "hardware": hardware_info(),
        **meta,
    }
    JsonlLogger(meta_path).write(record)
    return meta_path
