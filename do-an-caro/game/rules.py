"""Rules and win condition checking for Caro (Gomoku).

Supports:
- Vietnamese rules (default): 5 or more in a row, not blocked at both ends.
  Board boundary is not considered a block.
- Freestyle rules: 5 or more in a row wins unconditionally.
"""

from __future__ import annotations

import numpy as np

from game.constants import (
    DIRECTIONS,
    EMPTY,
    RULE_FREESTYLE,
    RULE_VIETNAM,
    WIN_STREAK,
)


def check_win_at(
    board: np.ndarray,
    row: int,
    col: int,
    rule: str = RULE_VIETNAM,
    win_streak: int = WIN_STREAK,
) -> bool:
    """Kiểm tra cục bộ xem nước đi tại (row, col) có tạo thành chuỗi thắng hay không.

    Độ phức tạp: O(1) - chỉ kiểm tra tối đa 4 hướng xung quanh ô vừa đánh.

    Args:
        board: Mảng 2D int8 biểu diễn bàn cờ.
        row: Chỉ số hàng của ô vừa đánh.
        col: Chỉ số cột của ô vừa đánh.
        rule: "vietnam" (mặc định) hoặc "freestyle".
        win_streak: Số quân tối thiểu liên tiếp để thắng (mặc định: 5).

    Returns:
        True nếu nước đi tạo thành chuỗi thắng, ngược lại False.
    """
    player = int(board[row, col])
    if player == EMPTY:
        return False

    size_r, size_c = board.shape
    opponent = -player

    for dr, dc in DIRECTIONS:
        # 1. Đếm theo hướng dương (+dr, +dc)
        len_pos = 0
        r, c = row + dr, col + dc
        while 0 <= r < size_r and 0 <= c < size_c and board[r, c] == player:
            len_pos += 1
            r += dr
            c += dc

        # Kiểm tra đầu dương có bị đối thủ chặn không
        # Biên bàn không tính là chặn
        blocked_pos = (
            0 <= r < size_r
            and 0 <= c < size_c
            and board[r, c] == opponent
        )

        # 2. Đếm theo hướng âm (-dr, -dc)
        len_neg = 0
        r, c = row - dr, col - dc
        while 0 <= r < size_r and 0 <= c < size_c and board[r, c] == player:
            len_neg += 1
            r -= dr
            c -= dc

        # Kiểm tra đầu âm có bị đối thủ chặn không
        # Biên bàn không tính là chặn
        blocked_neg = (
            0 <= r < size_r
            and 0 <= c < size_c
            and board[r, c] == opponent
        )

        # Tổng số quân liên tiếp
        total_len = 1 + len_pos + len_neg

        if total_len >= win_streak:
            if rule == RULE_FREESTYLE:
                return True
            # Luật Việt Nam: không bị chặn cả 2 đầu bởi quân đối thủ
            if not (blocked_pos and blocked_neg):
                return True

    return False


def check_win_global(
    board: np.ndarray,
    player: int,
    rule: str = RULE_VIETNAM,
    win_streak: int = WIN_STREAK,
) -> bool:
    """Quét toàn bộ bàn cờ để kiểm tra xem `player` có chuỗi thắng hay không.

    Hữu ích cho việc kiểm thử các thế cờ dựng sẵn hoặc kiểm tra tính nhất quán.

    Args:
        board: Mảng 2D int8 biểu diễn bàn cờ.
        player: Người chơi cần kiểm tra (BLACK=1 hoặc WHITE=-1).
        rule: "vietnam" hoặc "freestyle".
        win_streak: Số quân tối thiểu liên tiếp để thắng.

    Returns:
        True nếu player có chuỗi thắng, ngược lại False.
    """
    if player == EMPTY:
        return False

    size_r, size_c = board.shape
    opponent = -player

    for dr, dc in DIRECTIONS:
        for r in range(size_r):
            for c in range(size_c):
                if board[r, c] != player:
                    continue

                # Chỉ bắt đầu kiểm tra từ đầu chuỗi (ô phía trước không phải quân mình)
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < size_r and 0 <= prev_c < size_c and board[prev_r, prev_c] == player:
                    continue

                # Đếm độ dài chuỗi bắt đầu từ (r, c)
                curr_len = 0
                curr_r, curr_c = r, c
                while 0 <= curr_r < size_r and 0 <= curr_c < size_c and board[curr_r, curr_c] == player:
                    curr_len += 1
                    curr_r += dr
                    curr_c += dc

                if curr_len >= win_streak:
                    if rule == RULE_FREESTYLE:
                        return True

                    # Kiểm tra chặn 2 đầu
                    # Đầu trước (prev_r, prev_c)
                    blocked_start = (
                        0 <= prev_r < size_r
                        and 0 <= prev_c < size_c
                        and board[prev_r, prev_c] == opponent
                    )
                    # Đầu sau (curr_r, curr_c)
                    blocked_end = (
                        0 <= curr_r < size_r
                        and 0 <= curr_c < size_c
                        and board[curr_r, curr_c] == opponent
                    )

                    if not (blocked_start and blocked_end):
                        return True

    return False
