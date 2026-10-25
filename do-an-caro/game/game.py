"""Core Caro (Gomoku) game engine and unified API.

Implements all required methods:
- reset()
- legal_actions()
- step(action)
- is_terminal()
- result()
- render()
- clone()
- undo()
"""

from __future__ import annotations

import numpy as np

from game.constants import (
    DEFAULT_RULE,
    BLACK,
    DEFAULT_BOARD_SIZE,
    EMPTY,
    PLAYER_SYMBOLS,
    RULE_FREESTYLE,
    RULE_VIETNAM,
    WHITE,
    WIN_STREAK,
)
from game.rules import check_win_at


class CaroGame:
    """Môi trường game Caro bàn N×N với API thống nhất."""

    def __init__(
        self,
        size: int = DEFAULT_BOARD_SIZE,
        rule: str = DEFAULT_RULE,
        win_streak: int = WIN_STREAK,
    ) -> None:
        """Khởi tạo bàn cờ Caro.

        Args:
            size: Kích thước bàn cờ (mặc định: 11).
            rule: Luật kiểm tra thắng ("freestyle" - mặc định, hoặc "vietnam").
            win_streak: Số quân liên tiếp tối thiểu để thắng (mặc định: 5).
        """
        if size < win_streak:
            raise ValueError(f"Kích thước bàn cờ ({size}) không thể nhỏ hơn win_streak ({win_streak}).")
        if rule not in (RULE_VIETNAM, RULE_FREESTYLE):
            raise ValueError(f"Luật chơi '{rule}' không hợp lệ. Chọn '{RULE_VIETNAM}' hoặc '{RULE_FREESTYLE}'.")

        self.size = size
        self.num_cells = size * size
        self.rule = rule
        self.win_streak = win_streak

        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player: int = BLACK
        self.last_move: int | None = None
        self.winner: int | None = None
        self.is_terminal_flag: bool = False
        self.move_count: int = 0

        # Lịch sử phục vụ undo(): (action, prev_last_move, prev_winner, prev_terminal)
        self._history: list[tuple[int, int | None, int | None, bool]] = []

    def action_to_coord(self, action: int) -> tuple[int, int]:
        """Chuyển đổi chỉ số action (0..N^2-1) thành toạ độ (row, col)."""
        return action // self.size, action % self.size

    def coord_to_action(self, row: int, col: int) -> int:
        """Chuyển đổi toạ độ (row, col) thành chỉ số action (0..N^2-1)."""
        return row * self.size + col

    def reset(self) -> CaroGame:
        """Đặt lại bàn cờ về trạng thái ban đầu."""
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.last_move = None
        self.winner = None
        self.is_terminal_flag = False
        self.move_count = 0
        self._history.clear()
        return self

    def legal_actions(self) -> list[int]:
        """Trả về danh sách các action hợp lệ (các ô còn trống)."""
        if self.is_terminal_flag:
            return []
        # np.flatnonzero tìm các vị trí có giá trị EMPTY (0)
        return np.flatnonzero(self.board.ravel() == EMPTY).tolist()

    def step(self, action: int | tuple[int, int]) -> CaroGame:
        """Thực hiện một nước đi trên bàn cờ.

        Args:
            action: Chỉ số ô (0..size*size-1) hoặc tuple (row, col).

        Returns:
            self: Đối tượng bàn cờ sau khi đã thực hiện nước đi.

        Raises:
            RuntimeError: Nếu ván đấu đã kết thúc.
            ValueError: Nếu action ngoài phạm vi hoặc ô đã có quân.
        """
        if self.is_terminal_flag:
            raise RuntimeError("Ván cờ đã kết thúc, không thể thực hiện thêm nước đi.")

        # Chuẩn hoá action sang int index và toạ độ
        if isinstance(action, (tuple, list)):
            row, col = action
            action_idx = self.coord_to_action(row, col)
        else:
            action_idx = int(action)
            row, col = self.action_to_coord(action_idx)

        if not (0 <= action_idx < self.num_cells) or not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Action {action} nằm ngoài phạm vi bàn cờ (0..{self.num_cells - 1}).")

        if self.board[row, col] != EMPTY:
            raise ValueError(f"Ô tại toạ độ ({row}, {col}) (action={action_idx}) đã có quân cờ.")

        # Lưu lại lịch sử trước khi thay đổi trạng thái
        self._history.append((action_idx, self.last_move, self.winner, self.is_terminal_flag))

        # Đặt quân
        player = self.current_player
        self.board[row, col] = player
        self.last_move = action_idx
        self.move_count += 1

        # Kiểm tra thắng cục bộ O(1) quanh ô vừa đánh
        has_won = check_win_at(
            board=self.board,
            row=row,
            col=col,
            rule=self.rule,
            win_streak=self.win_streak,
        )

        if has_won:
            self.is_terminal_flag = True
            self.winner = player
        elif self.move_count == self.num_cells:
            # Bàn cờ đã đầy mà không ai thắng -> Hoà
            self.is_terminal_flag = True
            self.winner = 0
        else:
            # Đổi lượt
            self.current_player = -player

        return self

    def undo(self) -> CaroGame:
        """Thu hồi nước đi cuối cùng.

        Returns:
            self: Bàn cờ đã được khôi phục về trạng thái trước nước đi cuối.

        Raises:
            IndexError: Nếu không còn nước đi nào trong lịch sử để thu hồi.
        """
        if not self._history:
            raise IndexError("Không có nước đi nào trong lịch sử để thu hồi (undo).")

        action_idx, prev_last_move, prev_winner, prev_terminal = self._history.pop()
        row, col = self.action_to_coord(action_idx)

        player_who_moved = int(self.board[row, col])
        self.board[row, col] = EMPTY
        self.move_count -= 1
        self.current_player = player_who_moved
        self.last_move = prev_last_move
        self.winner = prev_winner
        self.is_terminal_flag = prev_terminal

        return self

    def is_terminal(self) -> bool:
        """Kiểm tra ván cờ đã kết thúc hay chưa (thắng/thua/hoà)."""
        return self.is_terminal_flag

    def result(self) -> int | None:
        """Trả về kết quả ván cờ.

        Returns:
            +1: X (BLACK) thắng
            -1: O (WHITE) thắng
             0: Hoà (bàn cờ đầy)
            None: Ván cờ chưa kết thúc
        """
        return self.winner

    def get_reward(self, player: int) -> float:
        """Trả về phần thưởng theo góc nhìn của người chơi `player`.

        +1.0 nếu thắng, -1.0 nếu thua, 0.0 nếu hoà hoặc chưa kết thúc.
        """
        if not self.is_terminal_flag or self.winner is None or self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def get_planes(self) -> np.ndarray:
        """Trả về biểu diễn 4 mặt phẳng 11×11 phục vụ mạng nơ-ron (SL/RL).

        Mặt phẳng 0: Quân của người chơi hiện tại (1.0 tại ô của mình, 0.0 nơi khác).
        Mặt phẳng 1: Quân của đối thủ (1.0 tại ô đối thủ, 0.0 nơi khác).
        Mặt phẳng 2: Nước vừa đi cuối cùng (1.0 tại last_move, 0.0 nơi khác).
        Mặt phẳng 3: Lượt đi (toàn 1.0 nếu người hiện tại là BLACK, toàn 0.0 nếu WHITE).

        Returns:
            np.ndarray shape (4, size, size), dtype float32.
        """
        planes = np.zeros((4, self.size, self.size), dtype=np.float32)
        player = self.current_player

        planes[0] = (self.board == player).astype(np.float32)
        planes[1] = (self.board == -player).astype(np.float32)

        if self.last_move is not None:
            r, c = self.action_to_coord(self.last_move)
            planes[2, r, c] = 1.0

        if player == BLACK:
            planes[3].fill(1.0)

        return planes

    def clone(self) -> CaroGame:
        """Tạo một bản sao độc lập hoàn chỉnh của game (phục vụ tìm kiếm MCTS/Minimax)."""
        copied = CaroGame.__new__(CaroGame)
        copied.size = self.size
        copied.num_cells = self.num_cells
        copied.rule = self.rule
        copied.win_streak = self.win_streak

        copied.board = self.board.copy()
        copied.current_player = self.current_player
        copied.last_move = self.last_move
        copied.winner = self.winner
        copied.is_terminal_flag = self.is_terminal_flag
        copied.move_count = self.move_count
        copied._history = list(self._history)
        return copied

    def render(self, mode: str = "text") -> str:
        """Hiển thị bàn cờ dạng chuỗi ký tự trực quan.

        Args:
            mode: "text" (trả về chuỗi) hoặc "print" (in ra màn hình và trả về chuỗi).
        """
        header_cols = "   " + " ".join(f"{c:2d}" for c in range(self.size))
        lines = [header_cols]

        for r in range(self.size):
            row_symbols = []
            for c in range(self.size):
                act = self.coord_to_action(r, c)
                sym = PLAYER_SYMBOLS[self.board[r, c]]
                if act == self.last_move:
                    # Đánh dấu nước đi vừa thực hiện
                    row_symbols.append(f"[{sym}]" if len(sym) == 1 else sym)
                else:
                    row_symbols.append(f" {sym} ")
            lines.append(f"{r:2d} " + "".join(row_symbols))

        status = f"Turn: {PLAYER_SYMBOLS[self.current_player]} ({self.current_player})"
        if self.is_terminal_flag:
            if self.winner == 0:
                status = "Result: Draw (Full board)!"
            else:
                status = f"Result: {PLAYER_SYMBOLS[self.winner]} wins!"
        lines.append(status)

        out = "\n".join(lines)
        if mode == "print":
            print(out)
        return out

    def __repr__(self) -> str:
        return (
            f"CaroGame(size={self.size}, rule='{self.rule}', "
            f"player={self.current_player}, moves={self.move_count}, "
            f"terminal={self.is_terminal_flag}, winner={self.winner})"
        )

    def __str__(self) -> str:
        return self.render(mode="text")
