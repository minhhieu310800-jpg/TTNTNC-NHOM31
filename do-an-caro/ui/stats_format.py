"""Định dạng ``last_stats`` của tác tử thành các dòng tiếng Việt để hiển thị."""

from __future__ import annotations

from typing import Any, Final

# Khóa last_stats -> nhãn hiển thị. Khóa lạ vẫn được hiển thị nguyên tên.
STAT_LABELS: Final[dict[str, str]] = {
    "nodes": "Số nút duyệt",
    "depth_reached": "Độ sâu đạt được",
    "cutoffs": "Số lần cắt tỉa",
    "tt_hits": "Lần trúng bảng chuyển vị",
    "score": "Điểm đánh giá",
    "sims": "Số mô phỏng",
    "best_visits": "Lượt thăm nước tốt nhất",
    "win_rate": "Tỷ lệ thắng ước tính",
    "rule": "Luật heuristic dùng",
    "time_used": "Thời gian suy nghĩ",
}


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return "không có"
    if key == "win_rate" and isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    if key == "time_used" and isinstance(value, (int, float)):
        return f"{value:.3f} giây"
    if isinstance(value, bool):
        return "có" if value else "không"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", " ")
    return str(value)


def format_stats(stats: dict[str, Any] | None, elapsed: float | None = None) -> list[str]:
    """Chuyển ``last_stats`` thành danh sách dòng "Nhãn: giá trị".

    Args:
        stats: ``last_stats`` của tác tử (có thể rỗng, như Random).
        elapsed: Thời gian thực tế (giây) của nước đi; dùng khi ``stats`` không có
            ``time_used`` để vẫn hiển thị được thời gian suy nghĩ.
    """
    lines: list[str] = []
    stats = dict(stats or {})
    if elapsed is not None and "time_used" not in stats:
        stats["time_used"] = elapsed
    for key, value in stats.items():
        label = STAT_LABELS.get(key, key)
        lines.append(f"{label}: {_format_value(key, value)}")
    return lines

