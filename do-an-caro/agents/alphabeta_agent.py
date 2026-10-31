"""Cấp 3: Minimax (negamax) + Alpha-Beta + sắp xếp nước + iterative deepening.

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


class SearchTimeout(Exception):
    """Ném ra khi hết thời gian suy nghĩ giữa chừng một độ sâu."""


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
        self._deadline: float | None = None

    def _get_moves(self, state: CaroGame) -> list[int]:
        """Nước ứng viên, sắp theo heuristic (cao -> thấp) nếu bật move ordering."""
        if self.use_move_ordering:
            scored = order_moves_by_heuristic(
                state, state.current_player, radius=self.radius
            )
            return [action for action, _ in scored]
        return get_candidate_moves(state.board, self.radius)

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """budget=None: tìm đúng độ sâu self.depth; budget=số: iterative deepening."""
        start = time.perf_counter()
        self._nodes = 0
        self._cutoffs = 0
        self._deadline = None

        if budget is None:
            best_action, best_value = self._search_root(state, self.depth)
            depth_reached = self.depth
        else:
            # Độ sâu 1 chạy không deadline để chắc chắn luôn có nước trả về.
            best_action, best_value = self._search_root(state, 1)
            depth_reached = 1
            self._deadline = start + budget
            max_depth = len(state.legal_actions())
            for d in range(2, max_depth + 1):
                try:
                    best_action, best_value = self._search_root(state, d)
                    depth_reached = d
                except SearchTimeout:
                    break
            self._deadline = None

        self.last_stats = {
            "nodes": self._nodes,
            "depth_reached": depth_reached,
            "cutoffs": self._cutoffs,
            "tt_hits": 0,
            "score": best_value,
            "time_used": time.perf_counter() - start,
        }
        return best_action

    def _search_root(self, state: CaroGame, depth: int) -> tuple[int, float]:
        """Tìm ở gốc với độ sâu `depth`, trả (nước tốt nhất, giá trị)."""
        best_action = -1
        best_value = -INF
        alpha = -INF
        for action in self._get_moves(state):
            state.step(action)
            try:
                value = -self._negamax(state, depth - 1, -INF, -alpha, ply=1)
            finally:
                state.undo()
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)
        return best_action, best_value

    def _negamax(self, state: CaroGame, depth: int, alpha: float, beta: float, ply: int) -> float:
        """Giá trị của thế cờ từ góc nhìn của người SẮP đi."""
        self._nodes += 1

        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise SearchTimeout

        if state.is_terminal():
            if state.result() == 0:
                return 0.0
            return -(WIN_SCORE - ply)

        if depth == 0:
            return evaluate_board(state.board, state.current_player, state.rule)

        best = -INF
        for action in self._get_moves(state):
            state.step(action)
            try:
                value = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
            finally:
                state.undo()
            best = max(best, value)
            alpha = max(alpha, value)
            if self.use_pruning and alpha >= beta:
                self._cutoffs += 1
                break
        return best