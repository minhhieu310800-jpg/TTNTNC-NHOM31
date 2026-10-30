"""Cấp 3: Minimax (negamax) độ sâu cố định, sẽ mở rộng thành Alpha-Beta.

Tham khảo: Russell & Norvig, AIMA 4th ed., chương 5 (adversarial search).
Hàm lượng giá và sinh nước ứng viên dùng lại từ game/patterns.py.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agents.base import Agent
from game.patterns import evaluate_board, get_candidate_moves

if TYPE_CHECKING:
    from game.game import CaroGame

WIN_SCORE = 10**9      # lớn hơn PATTERN_FIVE (100_000_000)
DEFAULT_DEPTH = 2
CANDIDATE_RADIUS = 2


class AlphaBetaAgent(Agent):
    """Tác tử Minimax/Alpha-Beta cấp 3."""

    name = "alphabeta"

    def __init__(self, depth: int = DEFAULT_DEPTH, radius: int = CANDIDATE_RADIUS) -> None:
        super().__init__()
        self.depth = depth
        self.radius = radius
        self._nodes = 0

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Chọn nước có giá trị negamax lớn nhất ở độ sâu self.depth."""
        start = time.perf_counter()
        self._nodes = 0

        best_action = -1
        best_value = float("-inf")
        for action in get_candidate_moves(state.board, self.radius):
            state.step(action)
            value = -self._negamax(state, self.depth - 1, ply=1)
            state.undo()
            if value > best_value:
                best_value = value
                best_action = action

        self.last_stats = {
            "nodes_visited": self._nodes,
            "depth_reached": self.depth,
            "time_used": time.perf_counter() - start,
        }
        return best_action

    def _negamax(self, state: CaroGame, depth: int, ply: int) -> float:
        """Giá trị của thế cờ từ góc nhìn của người SẮP đi."""
        self._nodes += 1

        if state.is_terminal():
            if state.result() == 0:
                return 0.0
            # Ván kết thúc có người thắng nghĩa là người VỪA đi đã thắng,
            # nên người sắp đi (ở nút này) là bên thua. Trừ ply để thích thắng sớm.
            return -(WIN_SCORE - ply)

        if depth == 0:
            return evaluate_board(state.board, state.current_player, state.rule)

        best = float("-inf")
        for action in get_candidate_moves(state.board, self.radius):
            state.step(action)
            best = max(best, -self._negamax(state, depth - 1, ply + 1))
            state.undo()
        return best