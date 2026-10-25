"""Pytest tests for Caro rules and win condition logic.

Covers:
1. Wins in all 4 directions: horizontal, vertical, main diagonal, anti-diagonal.
2. Overline: 6 or more in a row is a win.
3. Blocked ends rule:
   - Blocked at 1 end -> STILL WINS.
   - Blocked at 2 ends -> DOES NOT WIN.
4. Edge/border rule: Board edge does NOT count as blocked.
5. Freestyle rule: 5 in a row wins even if blocked at both ends.
6. Board size 9x9 generalization.
"""

import pytest
import numpy as np

from game.constants import (
    BLACK,
    DEFAULT_BOARD_SIZE,
    EMPTY,
    RULE_FREESTYLE,
    RULE_VIETNAM,
    WHITE,
)
from game.game import CaroGame
from game.rules import check_win_at, check_win_global


class TestWinFourDirections:
    """Kiểm tra điều kiện thắng ở 4 hướng."""

    def test_horizontal_win(self):
        """Thắng theo hàng ngang."""
        game = CaroGame(size=11, rule=RULE_VIETNAM)
        # X: (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)
        # O: (0, 0), (0, 1), (0, 2), (0, 3)
        moves = [
            (5, 3), (0, 0),
            (5, 4), (0, 1),
            (5, 5), (0, 2),
            (5, 6), (0, 3),
            (5, 7),  # Nước thắng của X
        ]
        for m in moves:
            game.step(m)

        assert game.is_terminal() is True
        assert game.result() == BLACK
        assert game.winner == BLACK

    def test_vertical_win(self):
        """Thắng theo cột dọc."""
        game = CaroGame(size=11, rule=RULE_VIETNAM)
        # X: (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)
        moves = [
            (2, 5), (0, 0),
            (3, 5), (0, 1),
            (4, 5), (0, 2),
            (5, 5), (0, 3),
            (6, 5),
        ]
        for m in moves:
            game.step(m)

        assert game.is_terminal() is True
        assert game.result() == BLACK

    def test_main_diagonal_win(self):
        """Thắng theo đường chéo chính (top-left -> bottom-right)."""
        game = CaroGame(size=11, rule=RULE_VIETNAM)
        # X: (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)
        moves = [
            (2, 2), (0, 1),
            (3, 3), (0, 2),
            (4, 4), (0, 3),
            (5, 5), (0, 4),
            (6, 6),
        ]
        for m in moves:
            game.step(m)

        assert game.is_terminal() is True
        assert game.result() == BLACK

    def test_anti_diagonal_win(self):
        """Thắng theo đường chéo phụ (top-right -> bottom-left)."""
        game = CaroGame(size=11, rule=RULE_VIETNAM)
        # X: (2, 8), (3, 7), (4, 6), (5, 5), (6, 4)
        moves = [
            (2, 8), (0, 0),
            (3, 7), (0, 1),
            (4, 6), (0, 2),
            (5, 5), (0, 3),
            (6, 4),
        ]
        for m in moves:
            game.step(m)

        assert game.is_terminal() is True
        assert game.result() == BLACK


class TestOverlineAndLongChains:
    """Kiểm tra chuỗi từ 6 quân liên tiếp trở lên (overline)."""

    def test_overline_six_stones_horizontal(self):
        """Chuỗi 6 quân liên tiếp theo hàng ngang vẫn thắng."""
        board = np.zeros((11, 11), dtype=np.int8)
        # Đặt 6 quân X: (5, 2) đến (5, 7)
        for c in range(2, 8):
            board[5, c] = BLACK

        # Nước đi cuối cùng nối thành 6 quân tại (5, 7)
        assert check_win_at(board, 5, 7, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_overline_six_stones_diagonal(self):
        """Chuỗi 6 quân liên tiếp theo đường chéo vẫn thắng."""
        board = np.zeros((11, 11), dtype=np.int8)
        for i in range(6):
            board[2 + i, 2 + i] = BLACK

        assert check_win_at(board, 7, 7, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_chain_seven_stones(self):
        """Chuỗi 7 quân liên tiếp vẫn thắng."""
        board = np.zeros((11, 11), dtype=np.int8)
        for r in range(1, 8):
            board[r, 5] = WHITE

        assert check_win_at(board, 7, 5, rule=RULE_VIETNAM) is True
        assert check_win_global(board, WHITE, rule=RULE_VIETNAM) is True


class TestBlockedEndsRule:
    """Kiểm tra luật chặn: chặn 1 đầu vẫn thắng, chặn 2 đầu không thắng."""

    def test_blocked_one_end_still_wins(self):
        """Chặn 1 đầu bởi đối thủ: O X X X X X . -> X VẪN THẮNG."""
        board = np.zeros((11, 11), dtype=np.int8)
        # O chặn ở (5, 2)
        board[5, 2] = WHITE
        # 5 quân X ở (5, 3) đến (5, 7)
        for c in range(3, 8):
            board[5, c] = BLACK
        # Ô (5, 8) trống

        assert check_win_at(board, 5, 7, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_blocked_two_ends_not_win(self):
        """Chặn cả 2 đầu bởi đối thủ: O X X X X X O -> X KHÔNG THẮNG."""
        board = np.zeros((11, 11), dtype=np.int8)
        # O chặn cả 2 đầu (5, 2) và (5, 8)
        board[5, 2] = WHITE
        board[5, 8] = WHITE
        # 5 quân X ở (5, 3) đến (5, 7)
        for c in range(3, 8):
            board[5, c] = BLACK

        # Nước đi cuối tại (5, 5) hoặc (5, 7)
        assert check_win_at(board, 5, 7, rule=RULE_VIETNAM) is False
        assert check_win_at(board, 5, 5, rule=RULE_VIETNAM) is False
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is False

    def test_blocked_two_ends_six_stones_not_win(self):
        """Chặn 2 đầu trên chuỗi 6 quân: O X X X X X X O -> KHÔNG THẮNG."""
        board = np.zeros((11, 11), dtype=np.int8)
        board[4, 1] = WHITE
        board[4, 8] = WHITE
        for c in range(2, 8):
            board[4, c] = BLACK

        assert check_win_at(board, 4, 7, rule=RULE_VIETNAM) is False
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is False

    def test_blocked_two_ends_game_continues(self):
        """Trong trận đấu thực tế, chuỗi bị chặn 2 đầu không làm kết thúc game."""
        game = CaroGame(size=11, rule=RULE_VIETNAM)
        # Dựng chuỗi: O ở (5, 2), X ở (5, 3..6), O ở (5, 8), rồi X đánh (5, 7) tạo O X X X X X O
        game.step((5, 3))  # X
        game.step((5, 2))  # O chặn đầu trái
        game.step((5, 4))  # X
        game.step((5, 8))  # O chặn đầu phải
        game.step((5, 5))  # X
        game.step((0, 0))  # O
        game.step((5, 6))  # X
        game.step((0, 1))  # O
        game.step((5, 7))  # X đánh tạo chuỗi 5 bị chặn 2 đầu!

        # Game KHÔNG được kết thúc
        assert game.is_terminal() is False
        assert game.result() is None
        # Đến lượt O
        assert game.current_player == WHITE

    def test_freestyle_rule_wins_even_if_blocked_two_ends(self):
        """Luật freestyle: 5 quân liên tiếp thắng ngay kể cả khi bị chặn 2 đầu."""
        board = np.zeros((11, 11), dtype=np.int8)
        board[5, 2] = WHITE
        board[5, 8] = WHITE
        for c in range(3, 8):
            board[5, c] = BLACK

        assert check_win_at(board, 5, 7, rule=RULE_FREESTYLE) is True
        assert check_win_global(board, BLACK, rule=RULE_FREESTYLE) is True


class TestEdgeAndCornerRules:
    """Kiểm tra thắng sát biên: Biên bàn KHÔNG tính là chặn."""

    def test_win_at_left_border(self):
        """Thắng sát biên trái: X ở cột 0..4, biên trái không tính là chặn."""
        board = np.zeros((11, 11), dtype=np.int8)
        # Cột 0 đến 4
        for c in range(5):
            board[5, c] = BLACK
        # Cột 5 trống

        assert check_win_at(board, 5, 0, rule=RULE_VIETNAM) is True
        assert check_win_at(board, 5, 4, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_win_at_right_border(self):
        """Thắng sát biên phải: X ở cột 6..10, biên phải không tính là chặn."""
        board = np.zeros((11, 11), dtype=np.int8)
        for c in range(6, 11):
            board[3, c] = BLACK

        assert check_win_at(board, 3, 10, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_win_touching_border_and_blocked_at_other_end(self):
        """Thắng sát biên 1 đầu, đầu kia bị đối thủ chặn: [Biên] X X X X X O.

        Biên không tính là chặn -> chỉ bị chặn 1 đầu -> VẪN THẮNG!
        """
        board = np.zeros((11, 11), dtype=np.int8)
        # X từ cột 0 đến 4
        for c in range(5):
            board[5, c] = BLACK
        # O chặn ở cột 5
        board[5, 5] = WHITE

        assert check_win_at(board, 5, 4, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_win_top_border_and_blocked_bottom(self):
        """Thắng sát biên trên theo cột dọc, đầu dưới bị chặn: [Biên] X X X X X O."""
        board = np.zeros((11, 11), dtype=np.int8)
        for r in range(5):
            board[r, 5] = BLACK
        board[5, 5] = WHITE

        assert check_win_at(board, 4, 5, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True

    def test_diagonal_starting_at_corner(self):
        """Đường chéo bắt đầu từ góc (0, 0) đến (4, 4)."""
        board = np.zeros((11, 11), dtype=np.int8)
        for i in range(5):
            board[i, i] = BLACK

        assert check_win_at(board, 0, 0, rule=RULE_VIETNAM) is True
        assert check_win_at(board, 4, 4, rule=RULE_VIETNAM) is True
        assert check_win_global(board, BLACK, rule=RULE_VIETNAM) is True


class TestGeneralizationConfig:
    """Kiểm tra tính tổng quát với cấu hình size=9 và win_streak."""

    def test_board_size_9(self):
        """Bàn 9x9 với luật Việt Nam chuẩn."""
        game = CaroGame(size=9, rule=RULE_VIETNAM)
        # X: (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)
        for c in range(2, 6):
            game.step((4, c))
            game.step((0, c))
        # Nước thứ 5 của X
        game.step((4, 6))

        assert game.is_terminal() is True
        assert game.result() == BLACK


def test_default_rule_is_freestyle():
    """Luật mặc định của đồ án là Gomoku tự do quốc tế."""
    from game.constants import DEFAULT_RULE

    assert DEFAULT_RULE == RULE_FREESTYLE
    assert CaroGame().rule == RULE_FREESTYLE

    # Chuỗi 5 bị chặn hai đầu vẫn thắng khi dùng luật mặc định
    board = np.zeros((11, 11), dtype=np.int8)
    board[5, 2] = WHITE
    board[5, 3:8] = BLACK
    board[5, 8] = WHITE
    assert check_win_at(board, 5, 5) is True
    assert check_win_global(board, BLACK) is True
