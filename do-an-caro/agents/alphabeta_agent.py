"""Cấp 3: Minimax (negamax) + cắt tỉa Alpha-Beta + sắp xếp nước, độ sâu cố định.

Tham khảo: Russell & Norvig, AIMA 4th ed., chương 5 (adversarial search).
Hàm lượng giá, sinh nước ứng viên và sắp nước dùng lại từ game/patterns.py.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agents.base import Agent
from game.patterns import (
    evaluate_board,
    get_candidate_moves,
    order_moves_by_heuristic,
)

if TYPE_CHECKING:
    from game.game import CaroGame

WIN_SCORE = 10**9      # lớn hơn PATTERN_FIVE (100_000_000)
DEFAULT_DEPTH = 2
CANDIDATE_RADIUS = 2
INF = float("inf")


class AlphaBetaAgent(Agent):
    """Tác tử Minimax/Alpha-Beta cấp 3."""

    name = "alphabeta"

    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        radius: int = CANDIDATE_RADIUS,
        use_pruning: bool = True,
        use_move_ordering: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.radius = radius
        self.use_pruning = use_pruning
        self.use_move_ordering = use_move_ordering
        self._nodes = 0
        self._cutoffs = 0

    def _get_moves(self, state: CaroGame) -> list[int]:
        """Nước ứng viên, sắp theo heuristic (cao -> thấp) nếu bật move ordering."""
        if self.use_move_ordering:
            scored = order_moves_by_heuristic(
                state, state.current_player, radius=self.radius
            )
            return [action for action, _ in scored]
        return get_candidate_moves(state.board, self.radius)

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Chọn nước có giá trị negamax lớn nhất ở độ sâu self.depth."""
        start = time.perf_counter()
        self._nodes = 0
        self._cutoffs = 0

        best_action = -1
        best_value = -INF
        alpha = -INF
        for action in self._get_moves(state):
            state.step(action)
            value = -self._negamax(state, self.depth - 1, -INF, -alpha, ply=1)
            state.undo()
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)

        self.last_stats = {
            "nodes": self._nodes,
            "depth_reached": self.depth,
            "cutoffs": self._cutoffs,
            "tt_hits": 0,
            "score": best_value,
            "time_used": time.perf_counter() - start,
        }
        return best_action

    def _negamax(self, state: CaroGame, depth: int, alpha: float, beta: float, ply: int) -> float:
        """Giá trị của thế cờ từ góc nhìn của người SẮP đi."""
        self._nodes += 1

        if state.is_terminal():
            if state.result() == 0:
                return 0.0
            return -(WIN_SCORE - ply)

        if depth == 0:
            return evaluate_board(state.board, state.current_player, state.rule)

        best = -INF
        for action in self._get_moves(state):
            state.step(action)
            value = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
            state.undo()
            best = max(best, value)
            alpha = max(alpha, value)
            if self.use_pruning and alpha >= beta:
                self._cutoffs += 1
                break
        return best