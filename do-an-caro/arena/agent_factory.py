"""Tạo tác tử từ cấu hình YAML để arena và worker đa tiến trình dùng chung."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent
from agents.base import Agent
from agents.puct_agent import PUCTAgent

# Loại tác tử -> lớp. Cấp 5–6 thêm vào đây khi có.
# "puct": cấp 7 (checkpoint mạng cấp 5/6) hoặc cấp 8 (checkpoint AlphaZero), params: checkpoint, simulations.
AGENT_TYPES: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "alphabeta": AlphaBetaAgent,
    "mcts": MCTSAgent,
    "puct": PUCTAgent,
}

# Tác tử có yếu tố ngẫu nhiên thật sự -> chạy thêm nhiều seed (proposal mục 6).
# Heuristic chỉ ngẫu nhiên khi hòa điểm nên coi là tất định như trong proposal.
STOCHASTIC_TYPES: set[str] = {"random", "mcts"}


@dataclass(frozen=True)
class AgentSpec:
    """Mô tả một tác tử trong giải đấu."""

    label: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    stochastic: bool = False

    @classmethod
    def from_config(cls, label: str, cfg: dict[str, Any]) -> AgentSpec:
        agent_type = cfg["type"]
        if agent_type not in AGENT_TYPES:
            raise ValueError(f"Loại tác tử '{agent_type}' chưa đăng ký. Có: {sorted(AGENT_TYPES)}")
        return cls(
            label=label,
            type=agent_type,
            params=dict(cfg.get("params") or {}),
            stochastic=bool(cfg.get("stochastic", agent_type in STOCHASTIC_TYPES)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "type": self.type, "params": dict(self.params)}


def build_agent(spec: AgentSpec, seed: int | None = None) -> Agent:
    """Tạo tác tử mới; truyền `seed` nếu lớp tác tử nhận tham số này."""
    cls = AGENT_TYPES[spec.type]
    kwargs = dict(spec.params)
    if seed is not None and "seed" in inspect.signature(cls.__init__).parameters:
        kwargs["seed"] = seed
    return cls(**kwargs)
