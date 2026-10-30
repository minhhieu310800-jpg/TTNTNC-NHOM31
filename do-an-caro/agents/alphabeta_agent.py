"""Cấp 3: Minimax (negamax) + cắt tỉa Alpha-Beta, độ sâu cố định.

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
INF = float("inf")


class AlphaBetaAgent(Agent):
    """Tác tử Minimax/Alpha-Beta cấp 3."""

    name = "alphabeta"

    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        radius: int = CANDIDATE_RADIUS,
        use_pruning: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.radius = radius
        self.use_pruning = use_pruning
        self._nodes = 0
        self._cutoffs = 0

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Chọn nước có giá trị negamax lớn nhất ở độ sâu self.depth."""
        start = time.perf_counter()
        self._nodes = 0
        self._cutoffs = 0

        best_action = -1
        best_value = -INF
        alpha = -INF  # giá trị tốt nhất đã tìm thấy ở gốc
        for action in get_candidate_moves(state.board, self.radius):
            state.step(action)
            # Cửa sổ của con là (-beta, -alpha) với beta = +INF ở gốc.
            value = -self._negamax(state, self.depth - 1, -INF, -alpha, ply=1)
            state.undo()
            if value > best_value:  # so sánh chặt: giữ nước đầu tiên khi bằng nhau
                best_value = value
                best_action = action
            alpha = max(alpha, value)  # ở gốc chỉ cập nhật alpha, không break

        self.last_stats = {
            "nodes_visited": self._nodes,
            "depth_reached": self.depth,
            "cutoffs": self._cutoffs,
            "tt_hits": 0,
            "score": best_value,
            "time_used": time.perf_counter() - start,
        }
        return best_action

    def _negamax(
        self, state: CaroGame, depth: int, alpha: float, beta: float, ply: int
    ) -> float:
        """Giá trị của thế cờ từ góc nhìn của người SẮP đi."""
        self._nodes += 1

        if state.is_terminal():
            if state.result() == 0:
                return 0.0
            # Có người thắng nghĩa là người VỪA đi đã thắng, nên người sắp đi
            # (ở nút này) là bên thua. Trừ ply để thích thắng sớm.
            return -(WIN_SCORE - ply)

        if depth == 0:
            return evaluate_board(state.board, state.current_player, state.rule)

        best = -INF
        for action in get_candidate_moves(state.board, self.radius):
            state.step(action)
            value = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
            state.undo()  # undo TRƯỚC khi break để bàn cờ luôn được khôi phục
            best = max(best, value)
            alpha = max(alpha, value)
            if self.use_pruning and alpha >= beta:
                self._cutoffs += 1
                break
        return best