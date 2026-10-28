"""Level 2 Agent: Rule-based Heuristic Agent for Caro 11x11.

Follows the exact priority chain specified in Issue #3:
1. Thắng ngay (Immediate Win)
2. Chặn thắng (Block Immediate Win)
3. Bốn mở (Create Open Four)
4. Chặn bốn / ba mở (Block Opponent's Four / Open Three)
5. Đòn kép (Create Fork / Block Opponent's Fork)
6. Điểm mẫu hình (Pattern Scoring & Center Control)

After each move, updates `last_stats = {"rule": tên_luật_vừa_dùng}`.
Never writes to file during action selection.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Final

from agents.base import Agent
from game.constants import EMPTY
from game.patterns import (
    evaluate_move,
    get_candidate_moves,
    is_fork_at,
    is_immediate_win,
    is_open_four_at,
)

if TYPE_CHECKING:
    from game.game import CaroGame

# Tên các luật theo đúng quy định issue #3
RULE_WIN: Final[str] = "thắng ngay"
RULE_BLOCK_WIN: Final[str] = "chặn thắng"
RULE_OPEN_FOUR: Final[str] = "bốn mở"
RULE_BLOCK_FOUR_OR_THREE: Final[str] = "chặn bốn/ba mở"
RULE_FORK: Final[str] = "đòn kép"
RULE_PATTERN_SCORE: Final[str] = "điểm mẫu hình"


class HeuristicAgent(Agent):
    """Tác tử cấp 2: Dựa trên chuỗi luật heuristic và đánh giá mẫu hình."""

    name: str = "heuristic"

    def __init__(self, seed: int | None = None, defense_weight: float = 1.0) -> None:
        """Khởi tạo HeuristicAgent.

        Args:
            seed: Seed cho bộ sinh số ngẫu nhiên (dùng khi hòa điểm mẫu hình).
            defense_weight: Trọng số ưu tiên phòng thủ so với tấn công.
        """
        super().__init__()
        self.seed = seed
        self.defense_weight = defense_weight
        self.rng = random.Random(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Đặt lại seed ngẫu nhiên cho tác tử."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Chọn nước đi dựa theo chuỗi ưu tiên luật heuristic.

        Chuỗi ưu tiên:
        thắng ngay → chặn thắng → bốn mở → chặn bốn/ba mở → đòn kép → điểm mẫu hình.

        Args:
            state: Trạng thái game hiện tại (không được sửa trực tiếp).
            budget: Thời gian tối đa (giây) - không bắt buộc với luật tĩnh.

        Returns:
            Chỉ số ô hợp lệ (0..size*size-1).
        """
        legal = state.legal_actions()
        if not legal:
            raise ValueError("Không còn nước đi hợp lệ nào trên bàn cờ.")

        # Nước đầu tiên trên bàn cờ trống hoàn toàn: đánh ngay trung tâm
        if len(legal) == state.num_cells:
            center = state.size // 2
            action = state.coord_to_action(center, center)
            self.last_stats = {"rule": RULE_PATTERN_SCORE}
            return action

        board = state.board
        player = state.current_player
        opponent = -player
        rule = state.rule
        win_streak = state.win_streak

        # Tập ứng viên trong bán kính 2 quanh các quân đã đánh
        candidate_set = set(get_candidate_moves(board, radius=2))
        candidates = [a for a in legal if a in candidate_set]
        if not candidates:
            candidates = legal

        # =====================================================================
        # 1. Thắng ngay: Nếu có nước tạo thành chuỗi thắng -> đánh ngay
        # =====================================================================
        for action in candidates:
            r, c = state.action_to_coord(action)
            if is_immediate_win(board, r, c, player, rule=rule, win_streak=win_streak):
                self.last_stats = {"rule": RULE_WIN}
                return action

        # =====================================================================
        # 2. Chặn thắng: Nếu đối thủ có nước thắng ngay ở lượt tới -> chặn ngay
        # =====================================================================
        opp_win_blocks: list[int] = []
        for action in candidates:
            r, c = state.action_to_coord(action)
            if is_immediate_win(board, r, c, opponent, rule=rule, win_streak=win_streak):
                opp_win_blocks.append(action)

        if opp_win_blocks:
            if len(opp_win_blocks) == 1:
                chosen = opp_win_blocks[0]
            else:
                chosen = max(
                    opp_win_blocks,
                    key=lambda a: evaluate_move(board, a, player, rule=rule, defense_weight=self.defense_weight),
                )
            self.last_stats = {"rule": RULE_BLOCK_WIN}
            return chosen

        # =====================================================================
        # 3. Bốn mở: Tạo chuỗi 4 quân hở 2 đầu (_XXXX_), đối thủ không thể cản
        # =====================================================================
        open_four_moves: list[int] = []
        for action in candidates:
            r, c = state.action_to_coord(action)
            if is_open_four_at(board, r, c, player, rule=rule):
                open_four_moves.append(action)

        if open_four_moves:
            chosen = max(
                open_four_moves,
                key=lambda a: evaluate_move(board, a, player, rule=rule, defense_weight=self.defense_weight),
            )
            self.last_stats = {"rule": RULE_OPEN_FOUR}
            return chosen

        # =====================================================================
        # 4. Chặn bốn / ba mở: Chặn đối thủ tạo bốn mở từ ba mở
        # =====================================================================
        opp_four_threats: list[int] = []
        for action in candidates:
            r, c = state.action_to_coord(action)
            if is_open_four_at(board, r, c, opponent, rule=rule):
                opp_four_threats.append(action)

        if opp_four_threats:
            chosen = max(
                opp_four_threats,
                key=lambda a: evaluate_move(board, a, player, rule=rule, defense_weight=self.defense_weight),
            )
            self.last_stats = {"rule": RULE_BLOCK_FOUR_OR_THREE}
            return chosen

        # =====================================================================
        # 5. Đòn kép: Tạo thế cờ có ít nhất 2 hướng đe dọa (3-3, 4-3, 4-4)
        #    hoặc chặn đòn kép của đối thủ
        # =====================================================================
        fork_moves: list[int] = []
        for action in candidates:
            r, c = state.action_to_coord(action)
            if is_fork_at(board, r, c, player, rule=rule):
                fork_moves.append(action)

        if fork_moves:
            chosen = max(
                fork_moves,
                key=lambda a: evaluate_move(board, a, player, rule=rule, defense_weight=self.defense_weight),
            )
            self.last_stats = {"rule": RULE_FORK}
            return chosen

        # Chặn đòn kép của đối thủ nếu đối thủ có nước tạo đòn kép
        opp_fork_moves: list[int] = []
        for action in candidates:
            r, c = state.action_to_coord(action)
            if is_fork_at(board, r, c, opponent, rule=rule):
                opp_fork_moves.append(action)

        if opp_fork_moves:
            chosen = max(
                opp_fork_moves,
                key=lambda a: evaluate_move(board, a, player, rule=rule, defense_weight=self.defense_weight),
            )
            self.last_stats = {"rule": RULE_FORK}
            return chosen

        # =====================================================================
        # 6. Điểm mẫu hình: Đánh giá toàn bộ điểm tấn công, phòng thủ và tâm
        # =====================================================================
        scored_moves: list[tuple[int, float]] = [
            (action, evaluate_move(board, action, player, rule=rule, defense_weight=self.defense_weight))
            for action in candidates
        ]

        max_score = max(score for _, score in scored_moves)
        best_candidates = [action for action, score in scored_moves if score >= max_score - 1e-6]

        if len(best_candidates) == 1:
            chosen = best_candidates[0]
        else:
            # Ưu tiên các ô gần tâm bàn cờ nhất
            center = state.size // 2
            dist_map = {
                a: abs(a // state.size - center) + abs(a % state.size - center)
                for a in best_candidates
            }
            min_dist = min(dist_map.values())
            closest_to_center = [a for a in best_candidates if dist_map[a] == min_dist]
            chosen = self.rng.choice(closest_to_center)

        self.last_stats = {"rule": RULE_PATTERN_SCORE}
        return chosen

    def __repr__(self) -> str:
        return f"HeuristicAgent(seed={self.seed}, defense_weight={self.defense_weight})"
