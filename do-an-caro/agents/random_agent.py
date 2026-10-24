"""Level 1 Agent: Random Agent with reproducible seed support."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agents.base import Agent

if TYPE_CHECKING:
    from game.game import CaroGame


class RandomAgent(Agent):
    """Tác tử cấp 1: Chọn ngẫu nhiên đều trong danh sách legal_actions()."""

    name: str = "random"

    def __init__(self, seed: int | None = None) -> None:
        """Khởi tạo RandomAgent với seed cố định hoặc ngẫu nhiên."""
        self.seed = seed
        self.rng = random.Random(seed)

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Chọn đều một nước đi từ state.legal_actions()."""
        legal = state.legal_actions()
        if not legal:
            raise ValueError("Không còn nước đi hợp lệ nào trên bàn cờ.")
        return self.rng.choice(legal)

    def reset_seed(self, seed: int | None = None) -> None:
        """Đặt lại seed cho bộ sinh số ngẫu nhiên."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

    def __repr__(self) -> str:
        return f"RandomAgent(seed={self.seed})"
