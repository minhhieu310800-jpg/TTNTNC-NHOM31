"""Cấp 3: Minimax (negamax) + Alpha-Beta + sắp xếp nước + iterative deepening.

Tham khảo: Russell & Norvig, AIMA 4th ed., chương 5 (adversarial search).
Hàm lượng giá, sinh nước ứng viên và sắp nước dùng lại từ game/patterns.py.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from agents.base import Agent
from agents.fast_eval import fast_evaluate
from game.patterns import (
    evaluate_board,
    get_candidate_moves,
    order_moves_by_heuristic,
)

if TYPE_CHECKING:
    from game.game import CaroGame

WIN_SCORE = 10**9      # lớn hơn PATTERN_FIVE (100_000_000)
MATE_SCORE_THRESHOLD = WIN_SCORE - 1_000
DEFAULT_DEPTH = 3
CANDIDATE_RADIUS = 2
INF = float("inf")
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
ZOBRIST_SEED = 20261120
ORDERING_HEURISTIC = "heuristic"   # order_moves_by_heuristic (chính xác, chậm)
ORDERING_CHEAP = "cheap"           # đếm quân lân cận (nhanh)


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
        ordering: str = ORDERING_CHEAP,
        fast_eval: bool = False,
        use_tt: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.radius = radius
        self.use_pruning = use_pruning
        self.use_move_ordering = use_move_ordering
        self._nodes = 0
        self._cutoffs = 0
        self.ordering = ordering
        self.fast_eval = fast_eval
        self._deadline: float | None = None
        self.use_tt = use_tt
        self._tt: dict[int, tuple[int, int, float, int | None]] = {}
        self._tt_hits = 0
        rng = np.random.default_rng(ZOBRIST_SEED)
        self._zobrist = rng.integers(0, 2**63, size=(2, 121), dtype=np.int64)
        self._side_key = int(rng.integers(0, 2**63, dtype=np.int64))

    def _hash(self, state: CaroGame) -> int:
        flat = state.board.ravel()
        h = 0
        for idx in np.flatnonzero(flat):
            h ^= int(self._zobrist[0 if flat[idx] == 1 else 1, idx])
        if state.current_player == 1:
            h ^= self._side_key
        return h

    @staticmethod
    def _score_to_tt(score: float, ply: int) -> float:
        if score >= MATE_SCORE_THRESHOLD:
            return score + ply
        if score <= -MATE_SCORE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: float, ply: int) -> float:
        if score >= MATE_SCORE_THRESHOLD:
            return score - ply
        if score <= -MATE_SCORE_THRESHOLD:
            return score + ply
        return score

    @staticmethod
    def _prioritize_move(moves: list[int], action: int | None) -> list[int]:
        if action is not None and action in moves:
            moves.remove(action)
            moves.insert(0, action)
        return moves

    def _cheap_order(self, state: CaroGame) -> list[int]:
        """Sắp ứng viên theo số quân lân cận (bán kính 1) và độ gần trung tâm."""
        board = state.board
        size = state.size
        occupied = (board != 0).astype(np.int32)
        padded = np.pad(occupied, 1)
        # Tổng quân trong ô 3x3 quanh mỗi ô.
        neighbors = sum(
            padded[1 + dr : 1 + dr + size, 1 + dc : 1 + dc + size]
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        )
        center = size // 2
        scored = []
        for action in get_candidate_moves(board, self.radius):
            r, c = divmod(action, size)
            score = 10 * int(neighbors[r, c]) - (abs(r - center) + abs(c - center))
            scored.append((score, action))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scored]

    def _get_moves(self, state: CaroGame) -> list[int]:
        if not self.use_move_ordering:
            return get_candidate_moves(state.board, self.radius)
        if self.ordering == ORDERING_CHEAP:
            return self._cheap_order(state)
        scored = order_moves_by_heuristic(state, state.current_player, radius=self.radius)
        return [action for action, _ in scored]

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """budget=None: tìm đúng độ sâu self.depth; budget=số: iterative deepening."""
        start = time.perf_counter()
        self._nodes = 0
        self._cutoffs = 0
        self._deadline = None
        self._tt_hits = 0
        self._tt.clear()

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
            "tt_hits": self._tt_hits,
            "score": best_value,
            "time_used": time.perf_counter() - start,
        }
        return best_action

    def _search_root(self, state: CaroGame, depth: int) -> tuple[int, float]:
        """Tìm ở gốc với độ sâu `depth`, trả (nước tốt nhất, giá trị)."""
        best_action = -1
        best_value = -INF
        alpha = -INF
        key = self._hash(state) if self.use_tt else 0
        entry = self._tt.get(key) if self.use_tt else None
        tt_move = entry[3] if entry is not None else None
        moves = self._prioritize_move(self._get_moves(state), tt_move)
        for action in moves:
            state.step(action)
            try:
                value = -self._negamax(state, depth - 1, -INF, -alpha, ply=1)
            finally:
                state.undo()
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)
        if self.use_tt:
            self._tt[key] = (
                depth,
                TT_EXACT,
                self._score_to_tt(best_value, 0),
                best_action,
            )
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
            if self.fast_eval:
                return fast_evaluate(state.board, state.current_player)
            return evaluate_board(state.board, state.current_player, state.rule)

        alpha_orig = alpha
        beta_orig = beta
        key = 0
        tt_move = None
        if self.use_tt:
            key = self._hash(state)
            entry = self._tt.get(key)
            if entry is not None:
                tt_move = entry[3]
            if entry is not None and entry[0] >= depth:
                _, flag, stored_value, _ = entry
                value = self._score_from_tt(stored_value, ply)
                if flag == TT_EXACT:
                    self._tt_hits += 1
                    return value
                if flag == TT_LOWER:
                    alpha = max(alpha, value)
                elif flag == TT_UPPER:
                    beta = min(beta, value)
                if alpha >= beta:
                    self._tt_hits += 1
                    return value

        best = -INF
        moves = self._get_moves(state) if depth >= 2 else get_candidate_moves(state.board, self.radius)
        moves = self._prioritize_move(moves, tt_move)
        best_action: int | None = None
        for action in moves:
            state.step(action)
            try:
                value = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
            finally:
                state.undo()
            if value > best:
                best = value
                best_action = action
            alpha = max(alpha, value)
            if self.use_pruning and alpha >= beta:
                self._cutoffs += 1
                break

        if self.use_tt:
            if best <= alpha_orig:
                flag = TT_UPPER
            elif best >= beta_orig:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt[key] = (
                depth,
                flag,
                self._score_to_tt(best, ply),
                best_action,
            )
        return best
