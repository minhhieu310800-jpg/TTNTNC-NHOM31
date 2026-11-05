"""Cấp 4: MCTS-UCT (Selection-Expansion-Simulation-Backpropagation).

Tham khảo: Kocsis & Szepesvari (2006); Russell & Norvig, AIMA 4th ed., ch. 5.
Rollout heuristic nhẹ: thắng ngay > chặn ngay > ngẫu nhiên trong vùng lân cận
nước vừa đi. Rollout bị cắt sau MAX_ROLLOUT_MOVES nước và chấm bằng fast_evaluate.
"""

from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING

from agents.base import Agent
from agents.fast_eval import fast_evaluate
from game.patterns import get_candidate_moves, is_immediate_win

if TYPE_CHECKING:
    from game.game import CaroGame

UCT_C = math.sqrt(2)
DEFAULT_SIMULATIONS = 1600
CANDIDATE_RADIUS = 2
LOCAL_RADIUS = 2            # vùng lân cận nước vừa đi trong rollout
MAX_ROLLOUT_MOVES = 24      # cắt rollout dài
EVAL_MARGIN = 500.0         # |điểm| nhỏ hơn mức này coi là hòa khi cắt rollout
ROLLOUT_RANDOM = "random"
ROLLOUT_HEURISTIC = "heuristic"


class Node:
    """Nút cây MCTS. `player` là người vừa đi để tới nút này."""

    __slots__ = ("parent", "action", "player", "children", "untried", "visits", "wins")

    def __init__(self, parent, action, player, untried):
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.untried = untried
        self.visits = 0
        self.wins = 0.0  # tổng thưởng theo góc nhìn của `player`

    def best_child(self, c: float) -> "Node":
        log_n = math.log(self.visits)
        return max(
            self.children,
            key=lambda ch: ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits),
        )


class MCTSAgent(Agent):
    """Tác tử MCTS-UCT cấp 4."""

    name = "mcts"

    def __init__(
        self,
        simulations: int = DEFAULT_SIMULATIONS,
        rollout: str = ROLLOUT_HEURISTIC,
        c: float = UCT_C,
        radius: int = CANDIDATE_RADIUS,
        max_rollout_moves: int = MAX_ROLLOUT_MOVES,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.simulations = simulations
        self.rollout = rollout
        self.c = c
        self.radius = radius
        self.max_rollout_moves = max_rollout_moves
        self.rng = random.Random(seed)

    def _forced_move(self, game: CaroGame) -> int | None:
        """Thắng ngay nếu có; nếu không thì chặn đối thủ thắng ngay."""
        me = game.current_player
        candidates = get_candidate_moves(game.board, self.radius)
        for player in (me, -me):
            for action in candidates:
                r, c = game.action_to_coord(action)
                if is_immediate_win(game.board, r, c, player, rule=game.rule):
                    return action
        return None

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """budget=None: self.simulations mô phỏng; budget=số: chạy theo giây."""
        start = time.perf_counter()
        game = state.clone()  # không bao giờ sửa state truyền vào

        forced = self._forced_move(game)
        if forced is not None:
            self.last_stats = {
                "sims": 0,
                "best_visits": 0,
                "win_rate": 1.0,
                "time_used": time.perf_counter() - start,
            }
            return forced

        deadline = None if budget is None else start + budget
        root = Node(None, None, -game.current_player, self._moves(game))
        sims = 0
        while True:
            if deadline is None:
                if sims >= self.simulations:
                    break
            elif sims > 0 and time.perf_counter() >= deadline:
                break
            self._simulate(root, game)
            sims += 1

        best = max(root.children, key=lambda ch: ch.visits)
        self.last_stats = {
            "sims": sims,
            "best_visits": best.visits,
            "win_rate": (best.wins / best.visits + 1.0) / 2.0,  # thang [0, 1]
            "time_used": time.perf_counter() - start,
        }
        return best.action

    def _moves(self, game: CaroGame) -> list[int]:
        if game.is_terminal():
            return []
        moves = get_candidate_moves(game.board, self.radius)
        self.rng.shuffle(moves)
        return moves

    def _local_moves(self, game: CaroGame) -> list[int]:
        """Ô trống trong bán kính LOCAL_RADIUS quanh nước vừa đi (rẻ hơn quét cả bàn)."""
        size = game.size
        if game.last_move is None:
            return [size // 2 * size + size // 2]
        r0, c0 = divmod(game.last_move, size)
        board = game.board
        moves = []
        for r in range(max(0, r0 - LOCAL_RADIUS), min(size, r0 + LOCAL_RADIUS + 1)):
            for c in range(max(0, c0 - LOCAL_RADIUS), min(size, c0 + LOCAL_RADIUS + 1)):
                if board[r, c] == 0:
                    moves.append(r * size + c)
        return moves or game.legal_actions()

    def _rollout_action(self, game: CaroGame) -> int:
        moves = self._local_moves(game)
        if self.rollout == ROLLOUT_HEURISTIC:
            me = game.current_player
            for player in (me, -me):
                for action in moves:
                    r, c = game.action_to_coord(action)
                    if is_immediate_win(game.board, r, c, player, rule=game.rule):
                        return action
        return self.rng.choice(moves)

    def _simulate(self, root: Node, game: CaroGame) -> None:
        """Một lần mô phỏng; step/undo trên `game` rồi trả về như cũ."""
        node = root
        steps = 0
        try:
            # 1. Selection
            while not node.untried and node.children:
                node = node.best_child(self.c)
                game.step(node.action)
                steps += 1
            # 2. Expansion
            if node.untried:
                action = node.untried.pop()
                player = game.current_player
                game.step(action)
                steps += 1
                child = Node(node, action, player, self._moves(game))
                node.children.append(child)
                node = child
            # 3. Simulation (cùng `game`, undo ở finally)
            rolled = 0
            while not game.is_terminal() and rolled < self.max_rollout_moves:
                game.step(self._rollout_action(game))
                steps += 1
                rolled += 1
            result = self._outcome(game)
        finally:
            for _ in range(steps):
                game.undo()
        # 4. Backpropagation: thưởng theo góc nhìn người vừa đi ở mỗi nút
        while node is not None:
            node.visits += 1
            if result != 0 and node.parent is not None:
                node.wins += 1.0 if result == node.player else -1.0
            node = node.parent

    @staticmethod
    def _outcome(game: CaroGame) -> int:
        """+1/-1/0 tuyệt đối; nếu chưa kết thúc thì chấm bằng fast_evaluate."""
        if game.is_terminal():
            return game.result()
        me = game.current_player
        score = fast_evaluate(game.board, me)
        if score > EVAL_MARGIN:
            return me
        if score < -EVAL_MARGIN:
            return -me
        return 0