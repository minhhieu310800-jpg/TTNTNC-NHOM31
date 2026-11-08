"""Danh sách cấp AI 1–8 cho giao diện và cách tạo tác tử từ từng cấp.

Giao diện không import trực tiếp từng tác tử mà tra cứu ``AGENT_TYPES`` của
``arena.agent_factory`` (nơi README ghi "Cấp 5–8 thêm vào đây khi có"). Khi một
thành viên đăng ký tác tử cấp 5–8 vào ``AGENT_TYPES`` bằng đúng khóa trong
``LEVELS`` thì cấp đó tự bật trên giao diện, không phải sửa code giao diện.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from agents.base import Agent
from arena.agent_factory import AGENT_TYPES, AgentSpec, build_agent


@dataclass(frozen=True)
class LevelInfo:
    """Thông tin một cấp AI hiển thị trên giao diện."""

    level: int
    title: str
    agent_type: str  # khóa trong AGENT_TYPES
    params: dict[str, Any] = field(default_factory=dict)


LEVELS: Final[tuple[LevelInfo, ...]] = (
    LevelInfo(1, "Cấp 1 - Random", "random"),
    LevelInfo(2, "Cấp 2 - Heuristic", "heuristic"),
    LevelInfo(3, "Cấp 3 - Alpha-Beta", "alphabeta"),
    LevelInfo(4, "Cấp 4 - MCTS-UCT", "mcts"),
    LevelInfo(5, "Cấp 5 - Mạng policy-value", "policy_net"),
    LevelInfo(6, "Cấp 6 - RL self-play", "rl_selfplay"),
    LevelInfo(7, "Cấp 7 - PUCT", "puct"),
    LevelInfo(8, "Cấp 8 - AlphaZero", "alphazero"),
)

# Thời gian suy nghĩ cho phép chọn (giây/nước), theo proposal mục RQ1.
BUDGET_CHOICES: Final[tuple[float, ...]] = (0.1, 0.5, 1.0)
DEFAULT_BUDGET: Final[float] = 0.5


def get_level(level: int) -> LevelInfo:
    """Trả về thông tin của cấp ``level`` (1–8)."""
    for info in LEVELS:
        if info.level == level:
            return info
    raise ValueError(f"Cấp AI không hợp lệ: {level}. Chọn từ 1 đến {len(LEVELS)}.")


def is_available(level: int) -> bool:
    """Cấp này đã có tác tử đăng ký trong ``AGENT_TYPES`` chưa."""
    return get_level(level).agent_type in AGENT_TYPES


def create_agent(level: int, seed: int | None = None) -> Agent:
    """Tạo tác tử của cấp ``level``; ném ``ValueError`` nếu cấp chưa có."""
    info = get_level(level)
    if info.agent_type not in AGENT_TYPES:
        raise ValueError(f"{info.title} chưa có trong dự án.")
    spec = AgentSpec(label=info.title, type=info.agent_type, params=dict(info.params))
    return build_agent(spec, seed=seed)

