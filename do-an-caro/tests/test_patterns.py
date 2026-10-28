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

    def test_heuristic_rollout_action(self):
        """heuristic_rollout_action chọn nước hợp lệ."""
        game = CaroGame(size=11)
        game.step((5, 5))
        act = heuristic_rollout_action(game, player=WHITE)
        assert act in game.legal_actions()
