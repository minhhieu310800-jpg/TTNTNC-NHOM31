"""Danh sách cấp AI 1–8 cho giao diện và cách tạo tác tử từ từng cấp.

Giao diện không import trực tiếp từng tác tử mà tra cứu ``AGENT_TYPES`` của
``arena.agent_factory`` (nơi README ghi "Cấp 5–8 thêm vào đây khi có") và kiểm
tra file checkpoint trên đĩa (với các cấp mạng học sâu 5–8). Khi một thành viên
đăng ký tác tử cấp 5–8 vào ``AGENT_TYPES`` và có file checkpoint tương ứng thì
cấp đó tự bật trên giao diện.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from agents.base import Agent

try:
    import arena.agent_factory as _factory
    from arena.agent_factory import AGENT_TYPES, AgentSpec, build_agent
except ModuleNotFoundError:
    _factory = None  # type: ignore[assignment]
    from agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent

    AGENT_TYPES: dict[str, type[Agent]] = {
        "random": RandomAgent,
        "heuristic": HeuristicAgent,
        "alphabeta": AlphaBetaAgent,
        "mcts": MCTSAgent,
    }

    @dataclass(frozen=True)
    class AgentSpec:  # type: ignore[no-redef]
        label: str
        type: str
        params: dict[str, Any] = field(default_factory=dict)

    def build_agent(spec: AgentSpec, seed: int | None = None) -> Agent:
        cls = AGENT_TYPES[spec.type]
        kwargs = dict(spec.params)
        if seed is not None and "seed" in inspect.signature(cls.__init__).parameters:
            kwargs["seed"] = seed
        return cls(**kwargs)

CARO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


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
    LevelInfo(
        5,
        "Cấp 5 - Mạng policy-value",
        "sl",
        {"checkpoint": "checkpoints/supervised-policy-value-network/best.pt"},
    ),
    LevelInfo(
        6,
        "Cấp 6 - RL self-play",
        "rl",
        {"checkpoint": "checkpoints/rl_selfplay/best.pt"},
    ),
    LevelInfo(
        7,
        "Cấp 7 - PUCT",
        "puct",
        {"checkpoint": "checkpoints/supervised-policy-value-network/best.pt"},
    ),
    LevelInfo(
        8,
        "Cấp 8 - AlphaZero",
        "puct",
        {"checkpoint": "checkpoints/alphazero2/best.pt"},
    ),
)

# Thời gian suy nghĩ cho phép chọn (giây/nước), theo proposal mục RQ1.
BUDGET_CHOICES: Final[tuple[float, ...]] = (0.1, 0.5, 1.0)
DEFAULT_BUDGET: Final[float] = 0.5


def _resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Tìm đường dẫn checkpoint theo thư mục hiện hành hoặc thư mục gốc ``do-an-caro``."""
    path = Path(checkpoint)
    if path.is_file():
        return path
    candidate = CARO_ROOT / path
    if candidate.is_file():
        return candidate
    return path


def get_level(level: int) -> LevelInfo:
    """Trả về thông tin của cấp ``level`` (1–8)."""
    for info in LEVELS:
        if info.level == level:
            return info
    raise ValueError(f"Cấp AI không hợp lệ: {level}. Chọn từ 1 đến {len(LEVELS)}.")


def is_available(level: int) -> bool:
    """Cấp này đã có tác tử đăng ký trong ``AGENT_TYPES`` và đủ file checkpoint chưa."""
    info = get_level(level)
    if info.agent_type not in AGENT_TYPES:
        return False
    checkpoint = info.params.get("checkpoint")
    if checkpoint is not None and not _resolve_checkpoint(checkpoint).is_file():
        return False
    return True


def create_agent(level: int, seed: int | None = None) -> Agent:
    """Tạo tác tử của cấp ``level``; ném ``ValueError`` nếu cấp chưa sẵn sàng."""
    info = get_level(level)
    if info.agent_type not in AGENT_TYPES:
        raise ValueError(f"{info.title} chưa có trong dự án.")
    params = dict(info.params)
    if "checkpoint" in params:
        resolved = _resolve_checkpoint(params["checkpoint"])
        if not resolved.is_file():
            raise ValueError(f"{info.title} thiếu file checkpoint: {params['checkpoint']}")
        params["checkpoint"] = str(resolved)
    cls = AGENT_TYPES[info.agent_type]
    if _factory is not None and _factory.AGENT_TYPES.get(info.agent_type) is cls:
        spec = AgentSpec(label=info.title, type=info.agent_type, params=params)
        return build_agent(spec, seed=seed)
    kwargs = dict(params)
    if seed is not None and "seed" in inspect.signature(cls.__init__).parameters:
        kwargs["seed"] = seed
    return cls(**kwargs)

