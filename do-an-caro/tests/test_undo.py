"""Pytest tests for CaroGame undo() functionality.

Covers:
- undo() single move
- undo() multiple moves in sequence back to empty board
- undo() after winning move reverts terminal state and winner
- undo() on empty history raises IndexError
- State consistency across step-undo-step cycles
"""

import pytest
import numpy as np

from game.constants import BLACK, EMPTY, WHITE
from game.game import CaroGame


class TestUndo:
    """Kiểm thử tính đúng đắn của phương thức undo()."""

    def test_undo_single_move(self):
        """Khôi phục trạng thái ban đầu sau 1 nước đi."""
        game = CaroGame(size=11)
        game.step(60)

        assert game.board[5, 5] == BLACK
        assert game.current_player == WHITE
        assert game.move_count == 1
        assert game.last_move == 60

        game.undo()

        assert game.board[5, 5] == EMPTY
        assert game.current_player == BLACK
        assert game.move_count == 0
        assert game.last_move is None
        assert game.is_terminal() is False
        assert len(game.legal_actions()) == 121

    def test_undo_multiple_moves(self):
        """Đi một chuỗi 10 nước, sau đó undo toàn bộ 10 nước về ban đầu."""
        game = CaroGame(size=11)
        moves = [10, 11, 22, 33, 44, 55, 66, 77, 88, 99]

        for m in moves:
            game.step(m)

        assert game.move_count == 10
        assert len(game.legal_actions()) == 111

        # Undo từng nước một
        for i in reversed(range(len(moves))):
            game.undo()
            assert game.move_count == i
            expected_last_move = moves[i - 1] if i > 0 else None
            assert game.last_move == expected_last_move

        assert game.move_count == 0
        assert game.current_player == BLACK
        assert game.last_move is None
        assert np.all(game.board == EMPTY)
        assert len(game.legal_actions()) == 121

    def test_undo_after_win(self):
        """undo() sau nước đi chiến thắng phải đưa game trở lại trạng thái chưa kết thúc."""
        game = CaroGame(size=11)
        # Chuỗi 5 nước của X
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
        assert game.winner == BLACK
        assert game.result() == BLACK

        # Thu hồi nước thắng cuối cùng
        game.undo()

        assert game.is_terminal() is False
        assert game.winner is None
        assert game.result() is None
        assert game.board[5, 7] == EMPTY
        assert game.current_player == BLACK  # Vẫn là lượt của X để đi lại
        assert game.last_move == game.coord_to_action(0, 3)

    def test_undo_empty_history_raises_index_error(self):
        """Gọi undo() khi chưa có nước đi nào phải ném IndexError."""
        game = CaroGame(size=11)
        with pytest.raises(IndexError, match="Không có nước đi nào"):
            game.undo()

    def test_undo_replay_consistency(self):
        """Chơi A, B, C; undo C; chơi lại C -> trạng thái hoàn toàn đồng nhất."""
        game = CaroGame(size=11)
        game.step(10)
        game.step(20)

        # Lưu lại trạng thái trước khi đi nước thứ 3
        snapshot_board = game.board.copy()
        snapshot_player = game.current_player
        snapshot_last_move = game.last_move

        # Đi nước 30 rồi undo
        game.step(30)
        game.undo()

        assert np.array_equal(game.board, snapshot_board)
        assert game.current_player == snapshot_player
        assert game.last_move == snapshot_last_move
        assert game.move_count == 2
