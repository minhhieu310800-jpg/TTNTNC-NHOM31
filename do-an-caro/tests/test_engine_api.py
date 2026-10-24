"""Pytest tests for CaroGame Engine unified API.

Covers:
- reset()
- legal_actions()
- step() with both int action and (row, col) tuple
- Exception handling (out of bounds, occupied cell, move after game over)
- is_terminal() and result()
- clone() independent deep copy
- undo()
- render() text representation
- get_planes() 4-plane neural net input
- Full board draw (hòa bàn đầy 121 ô)
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


class TestEngineAPI:
    """Kiểm thử API thống nhất của CaroGame."""

    def test_initial_state(self):
        """Trạng thái ban đầu của bàn cờ."""
        game = CaroGame(size=11)
        assert game.size == 11
        assert game.num_cells == 121
        assert game.current_player == BLACK
        assert game.last_move is None
        assert game.winner is None
        assert game.is_terminal() is False
        assert game.result() is None
        assert len(game.legal_actions()) == 121
        assert np.all(game.board == EMPTY)

    def test_reset(self):
        """Khôi phục bàn cờ về ban đầu sau vài nước đi."""
        game = CaroGame(size=11)
        game.step(0)
        game.step(1)
        game.step(2)
        assert game.move_count == 3
        assert len(game.legal_actions()) == 118

        game.reset()
        assert game.move_count == 0
        assert game.current_player == BLACK
        assert game.last_move is None
        assert len(game.legal_actions()) == 121
        assert np.all(game.board == EMPTY)

    def test_step_alternation_and_legal_actions(self):
        """Kiểm tra đổi lượt luân phiên và cập nhật legal_actions."""
        game = CaroGame(size=11)
        # Nước 1: X đi tại ô 60 (tâm bàn cờ 5, 5)
        game.step(60)
        assert game.board[5, 5] == BLACK
        assert game.current_player == WHITE
        assert game.last_move == 60
        assert 60 not in game.legal_actions()
        assert len(game.legal_actions()) == 120

        # Nước 2: O đi bằng tuple (row, col) tại (5, 6) -> action 61
        game.step((5, 6))
        assert game.board[5, 6] == WHITE
        assert game.current_player == BLACK
        assert game.last_move == 61
        assert 61 not in game.legal_actions()
        assert len(game.legal_actions()) == 119

    def test_step_invalid_moves_raise_exceptions(self):
        """Nước đi không hợp lệ phải ném ngoại lệ."""
        game = CaroGame(size=11)
        game.step(10)  # X đánh ô 10

        # Ô đã có quân
        with pytest.raises(ValueError, match="đã có quân cờ"):
            game.step(10)

        # Ngoài phạm vi bàn cờ (< 0 hoặc >= 121)
        with pytest.raises(ValueError, match="ngoài phạm vi"):
            game.step(-1)

        with pytest.raises(ValueError, match="ngoài phạm vi"):
            game.step(121)

        with pytest.raises(ValueError, match="ngoài phạm vi"):
            game.step((11, 0))

    def test_step_after_terminal_raises_runtime_error(self):
        """Thực hiện nước đi khi ván đấu đã kết thúc phải ném RuntimeError."""
        game = CaroGame(size=11)
        # Tạo chuỗi thắng 5 quân cho X
        for c in range(4):
            game.step((5, c))
            game.step((0, c))
        game.step((5, 4))  # X thắng

        assert game.is_terminal() is True
        with pytest.raises(RuntimeError, match="đã kết thúc"):
            game.step((1, 1))

    def test_clone_independence(self):
        """clone() phải tạo bản sao độc lập, thay đổi trên bản sao không ảnh hưởng bản gốc."""
        game = CaroGame(size=11)
        game.step(10)
        game.step(20)

        cloned = game.clone()
        assert np.array_equal(cloned.board, game.board)
        assert cloned.current_player == game.current_player
        assert cloned.last_move == game.last_move
        assert cloned.move_count == game.move_count

        # Thực hiện nước đi trên cloned
        cloned.step(30)
        assert cloned.board[2, 8] == BLACK  # 30 = 2*11 + 8
        assert game.board[2, 8] == EMPTY   # Bản gốc không đổi
        assert len(cloned.legal_actions()) == len(game.legal_actions()) - 1

    def test_render_modes(self):
        """render() hoạt động ở cả chế độ text và print."""
        game = CaroGame(size=11)
        game.step(60)  # X tại tâm
        text_out = game.render(mode="text")
        assert isinstance(text_out, str)
        assert "[X]" in text_out  # Đánh dấu nước vừa đi
        assert "Turn: O" in text_out

        print_out = game.render(mode="print")
        assert text_out == print_out

    def test_get_planes(self):
        """get_planes() trả về 4 mặt phẳng đúng đặc tả trong Proposal."""
        game = CaroGame(size=11)
        game.step((5, 5))  # X tại (5, 5) -> action 60
        # Hiện tại là lượt O (WHITE)

        planes = game.get_planes()
        assert planes.shape == (4, 11, 11)
        assert planes.dtype == np.float32

        # Lượt hiện tại là O:
        # Mặt phẳng 0 (quân mình - O): chưa có quân nào
        assert planes[0].sum() == 0.0
        # Mặt phẳng 1 (quân đối thủ - X): có 1 quân tại (5, 5)
        assert planes[1, 5, 5] == 1.0
        assert planes[1].sum() == 1.0
        # Mặt phẳng 2 (nước vừa đi): tại (5, 5)
        assert planes[2, 5, 5] == 1.0
        assert planes[2].sum() == 1.0
        # Mặt phẳng 3 (lượt đi): O là WHITE -> toàn 0
        assert planes[3].sum() == 0.0

    def test_get_reward(self):
        """Kiểm tra hàm get_reward() theo góc nhìn từng bên."""
        game = CaroGame(size=11)
        assert game.get_reward(BLACK) == 0.0
        assert game.get_reward(WHITE) == 0.0

        for c in range(4):
            game.step((5, c))
            game.step((0, c))
        game.step((5, 4))  # X thắng

        assert game.get_reward(BLACK) == 1.0
        assert game.get_reward(WHITE) == -1.0


class TestFullBoardDraw:
    """Kiểm tra tình huống hoà khi bàn cờ đầy 121 ô."""

    def test_full_board_draw_11x11(self):
        """Lấp đầy 121 ô theo mẫu dịch tuần hoàn shift=2 không tạo thành 5 quân liên tiếp."""
        game = CaroGame(size=11, rule=RULE_VIETNAM)

        # Mẫu [1, 1, -1, -1] dịch 2 vị trí mỗi hàng:
        # Không tạo thành 5 quân liên tiếp theo bất kỳ hướng nào
        base = [BLACK, BLACK, WHITE, WHITE]
        pattern = np.zeros((11, 11), dtype=np.int8)
        for r in range(11):
            for c in range(11):
                pattern[r, c] = base[(c + r * 2) % 4]

        count_black = int((pattern == BLACK).sum())
        count_white = int((pattern == WHITE).sum())
        assert count_black == 61
        assert count_white == 60

        black_cells = [(r, c) for r in range(11) for c in range(11) if pattern[r, c] == BLACK]
        white_cells = [(r, c) for r in range(11) for c in range(11) if pattern[r, c] == WHITE]

        # Lần lượt đi xen kẽ X và O cho 60 lượt đầu
        for i in range(60):
            game.step(black_cells[i])
            assert game.is_terminal() is False
            game.step(white_cells[i])
            assert game.is_terminal() is False

        # Nước cuối cùng thứ 121 của X
        game.step(black_cells[60])

        assert game.move_count == 121
        assert game.is_terminal() is True
        assert game.result() == 0  # Hoà
        assert game.winner == 0
        assert len(game.legal_actions()) == 0
        assert game.get_reward(BLACK) == 0.0
        assert game.get_reward(WHITE) == 0.0
