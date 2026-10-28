"""Pytest tests for shared pattern module game/patterns.py."""

import numpy as np
import pytest

from game.constants import (
    BLACK,
    DEFAULT_BOARD_SIZE,
    EMPTY,
    RULE_FREESTYLE,
    RULE_VIETNAM,
    WHITE,
)
from game.game import CaroGame
from game.patterns import (
    PATTERN_BLOCKED_FOUR,
    PATTERN_FIVE,
    PATTERN_OPEN_FOUR,
    PATTERN_OPEN_THREE,
    detect_patterns_at,
    evaluate_board,
    evaluate_move,
    get_candidate_moves,
    heuristic_rollout_action,
    is_fork_at,
    is_immediate_win,
    is_open_four_at,
    order_moves_by_heuristic,
    score_point_patterns,
)


class TestCandidateMoves:
    """Kiểm thử sinh nước đi ứng viên."""

    def test_candidate_moves_empty_board(self):
        """Bàn cờ trống trả về duy nhất ô trung tâm."""
        board = np.zeros((11, 11), dtype=np.int8)
        cands = get_candidate_moves(board, radius=2)
        assert cands == [5 * 11 + 5]

    def test_candidate_moves_single_stone(self):
        """Quân ở trung tâm (5, 5), ứng viên là các ô trống trong bán kính 2."""
        board = np.zeros((11, 11), dtype=np.int8)
        board[5, 5] = BLACK
        cands = get_candidate_moves(board, radius=2)
        assert len(cands) == 24  # (5x5 vùng) - 1 ô đã có quân = 24
        assert (5 * 11 + 5) not in cands


class TestImmediateWin:
    """Kiểm thử phát hiện nước thắng ngay."""

    def test_immediate_win_horizontal(self):
        """4 quân liên tiếp -> ô thứ 5 thắng ngay."""
        board = np.zeros((11, 11), dtype=np.int8)
        for c in range(2, 6):
            board[5, c] = BLACK
        # Ô (5, 6) và (5, 1) là nước thắng ngay
        assert is_immediate_win(board, 5, 6, BLACK) is True
        assert is_immediate_win(board, 5, 1, BLACK) is True
        assert is_immediate_win(board, 5, 7, BLACK) is False

    def test_immediate_win_vietnam_blocked_both_ends(self):
        """Luật Việt Nam: bị chặn 2 đầu thì không thắng."""
        board = np.zeros((11, 11), dtype=np.int8)
        board[5, 1] = WHITE
        for c in range(2, 6):
            board[5, c] = BLACK
        board[5, 7] = WHITE
        # Ô (5, 6) đánh vào sẽ tạo chuỗi 5 bị O chặn ở (5, 1) và (5, 7)
        assert is_immediate_win(board, 5, 6, BLACK, rule=RULE_VIETNAM) is False
        assert is_immediate_win(board, 5, 6, BLACK, rule=RULE_FREESTYLE) is True


class TestOpenFourAndFork:
    """Kiểm thử bốn mở và đòn kép."""

    def test_open_four_detection(self):
        """3 quân liên tiếp có 2 đầu trống -> đặt thêm quân tạo Bốn Mở."""
        board = np.zeros((11, 11), dtype=np.int8)
        # _ X X X _
        board[5, 4] = BLACK
        board[5, 5] = BLACK
        board[5, 6] = BLACK
        # Đặt vào (5, 3) hoặc (5, 7) tạo thành Bốn Mở _ X X X X _
        assert is_open_four_at(board, 5, 3, BLACK) is True
        assert is_open_four_at(board, 5, 7, BLACK) is True
        assert is_open_four_at(board, 0, 0, BLACK) is False

    def test_fork_three_three(self):
        """Tạo đòn kép 3-3 (Double Open Three)."""
        board = np.zeros((11, 11), dtype=np.int8)
        # Tạo hàng ngang 2 quân và hàng dọc 2 quân giao nhau tại (5, 5)
        # Ngang: (5, 4), [5, 5], (5, 6) -> 3 mở
        # Dọc: (4, 5), [5, 5], (6, 5) -> 3 mở
        board[5, 4] = BLACK
        board[5, 6] = BLACK
        board[4, 5] = BLACK
        board[6, 5] = BLACK
        # Ô giao (5, 5) tạo thành 2 ba mở
        assert is_fork_at(board, 5, 5, BLACK) is True

    def test_fork_four_three(self):
        """Tạo đòn kép 4-3 (Một bốn và một ba mở)."""
        board = np.zeros((11, 11), dtype=np.int8)
        # Ngang có 3 quân: (5, 2), (5, 3), (5, 4) -> đặt (5, 5) thành bốn
        board[5, 2] = BLACK
        board[5, 3] = BLACK
        board[5, 4] = BLACK
        # Dọc có 2 quân: (4, 5), (6, 5) -> đặt (5, 5) thành ba mở
        board[4, 5] = BLACK
        board[6, 5] = BLACK
        assert is_fork_at(board, 5, 5, BLACK) is True


class TestGappedPatterns:
    """Kiểm thử các mẫu hình có khe (Review điểm 2 của nhóm trưởng)."""

    def test_gapped_four_xxx_x(self):
        """Bốn có khe dạng XXX_X tính là bốn chặn."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 2), (5, 3), (5, 4) [trống 5, 5] (5, 6)
        board[5, 2] = BLACK
        board[5, 3] = BLACK
        board[5, 4] = BLACK
        board[5, 6] = BLACK
        counts = detect_patterns_at(board, 5, 3, BLACK)
        assert counts.get(PATTERN_BLOCKED_FOUR, 0) >= 1

    def test_gapped_four_xx_xx(self):
        """Bốn có khe dạng XX_XX tính là bốn chặn."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 2), (5, 3) [trống 5, 4] (5, 5), (5, 6)
        board[5, 2] = BLACK
        board[5, 3] = BLACK
        board[5, 5] = BLACK
        board[5, 6] = BLACK
        counts = detect_patterns_at(board, 5, 3, BLACK)
        assert counts.get(PATTERN_BLOCKED_FOUR, 0) >= 1

    def test_gapped_four_x_xxx(self):
        """Bốn có khe dạng X_XXX tính là bốn chặn."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 2) [trống 5, 3] (5, 4), (5, 5), (5, 6)
        board[5, 2] = BLACK
        board[5, 4] = BLACK
        board[5, 5] = BLACK
        board[5, 6] = BLACK
        counts = detect_patterns_at(board, 5, 5, BLACK)
        assert counts.get(PATTERN_BLOCKED_FOUR, 0) >= 1

    def test_gapped_open_three_xx_x(self):
        """Ba có khe dạng _XX_X_ tính là ba mở."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 1) trống, (5, 2), (5, 3) X, (5, 4) trống, (5, 5) X, (5, 6) trống
        board[5, 2] = BLACK
        board[5, 3] = BLACK
        board[5, 5] = BLACK
        counts = detect_patterns_at(board, 5, 2, BLACK)
        assert counts.get(PATTERN_OPEN_THREE, 0) >= 1

    def test_gapped_open_three_x_xx(self):
        """Ba có khe dạng _X_XX_ tính là ba mở."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 1) trống, (5, 2) X, (5, 3) trống, (5, 4), (5, 5) X, (5, 6) trống
        board[5, 2] = BLACK
        board[5, 4] = BLACK
        board[5, 5] = BLACK
        counts = detect_patterns_at(board, 5, 4, BLACK)
        assert counts.get(PATTERN_OPEN_THREE, 0) >= 1

    def test_fork_with_gapped_three_and_contiguous_three(self):
        """Đòn kép kết hợp ba dọc liền và ba ngang có khe (Ví dụ của nhóm trưởng).

        Dọc liền: (4, 5), (6, 5)
        Ngang có khe: (5, 3), (5, 6)
        Đặt tại (5, 5) sẽ tạo thành:
        - Dọc: (4, 5), [5, 5], (6, 5) -> ba dọc liền (_XXX_)
        - Ngang: (5, 3), [5, 5], (5, 6) -> ba ngang có khe (_X_XX_ hoặc _XX_X_)
        -> is_fork_at((5, 5)) phải trả về True!
        """
        board = np.zeros((11, 11), dtype=np.int8)
        # Dọc: (4, 5), (6, 5)
        board[4, 5] = BLACK
        board[6, 5] = BLACK
        # Ngang có khe: (5, 3) và (5, 6)
        board[5, 3] = BLACK
        board[5, 6] = BLACK
        # Ô (5, 5) đang trống
        assert is_fork_at(board, 5, 5, BLACK) is True


class TestPatternScoringAndOrdering:
    """Kiểm thử hàm lượng giá và sắp xếp nước đi."""

    def test_evaluate_move_prioritizes_threats(self):
        """Nước thắng điểm cao hơn nước thường."""
        board = np.zeros((11, 11), dtype=np.int8)
        for c in range(3, 7):
            board[5, c] = BLACK
        win_score = evaluate_move(board, 5 * 11 + 7, BLACK)
        random_score = evaluate_move(board, 0, BLACK)
        assert win_score > random_score

    def test_order_moves_by_heuristic(self):
        """order_moves_by_heuristic trả về danh sách giảm dần theo điểm."""
        game = CaroGame(size=11)
        game.step((5, 5))
        ordered = order_moves_by_heuristic(game, player=WHITE)
        assert len(ordered) > 0
        scores = [score for _, score in ordered]
        assert scores == sorted(scores, reverse=True)

    def test_evaluate_board_symmetry(self):
        """evaluate_board trả về điểm dương cho bên ưu thế."""
        board = np.zeros((11, 11), dtype=np.int8)
        for c in range(3, 7):
            board[5, c] = BLACK
        # BLACK có 4 quân, WHITE chưa có gì -> điểm BLACK dương, điểm WHITE âm
        assert evaluate_board(board, BLACK) > 0
        assert evaluate_board(board, WHITE) < 0

    def test_evaluate_board_single_open_three_score(self):
        """1 ba mở duy nhất trên bàn cờ có điểm đúng bằng 50.000 (Review điểm 3)."""
        board = np.zeros((11, 11), dtype=np.int8)
        # _ X X X _
        board[5, 4] = BLACK
        board[5, 5] = BLACK
        board[5, 6] = BLACK
        assert evaluate_board(board, BLACK) == 50_000.0
        assert evaluate_board(board, WHITE) == -50_000.0

    def test_evaluate_board_single_blocked_four_score(self):
        """1 bốn chặn duy nhất trên bàn cờ có điểm đúng bằng 100.000 (Review điểm 3)."""
        board = np.zeros((11, 11), dtype=np.int8)
        # O X X X X _
        board[5, 2] = WHITE
        board[5, 3] = BLACK
        board[5, 4] = BLACK
        board[5, 5] = BLACK
        board[5, 6] = BLACK
        assert evaluate_board(board, BLACK) == 100_000.0
        assert evaluate_board(board, WHITE) == -100_000.0

    def test_evaluate_board_gapped_open_three_score(self):
        """1 ba mở có khe (_XX_X_) có điểm đúng bằng 50.000."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 2) X, (5, 3) X, (5, 5) X
        board[5, 2] = BLACK
        board[5, 3] = BLACK
        board[5, 5] = BLACK
        assert evaluate_board(board, BLACK) == 50_000.0
        assert evaluate_board(board, WHITE) == -50_000.0

    def test_evaluate_board_gapped_four_score(self):
        """1 bốn có khe (XXX_X) có điểm đúng bằng 100.000."""
        board = np.zeros((11, 11), dtype=np.int8)
        # (5, 2) X, (5, 3) X, (5, 4) X, (5, 6) X
        board[5, 2] = BLACK
        board[5, 3] = BLACK
        board[5, 4] = BLACK
        board[5, 6] = BLACK
        assert evaluate_board(board, BLACK) == 100_000.0
        assert evaluate_board(board, WHITE) == -100_000.0

    def test_heuristic_rollout_action(self):
        """heuristic_rollout_action chọn nước hợp lệ."""
        game = CaroGame(size=11)
        game.step((5, 5))
        act = heuristic_rollout_action(game, player=WHITE)
        assert act in game.legal_actions()
