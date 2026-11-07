"""Hàm lượng giá nhanh (Numba) cho Alpha-Beta và rollout MCTS.

Quét mọi cửa sổ 5 ô theo 4 hướng; cửa sổ chỉ chứa quân một bên thì cộng điểm
theo số quân và số đầu mở. Đúng luật Việt Nam: năm bị chặn cả 2 đầu không
được điểm thắng. Thang điểm lấy từ PATTERN_SCORES của game/patterns.py.

Xấp xỉ có chủ đích: một mẫu hình có thể được tính ở nhiều cửa sổ chồng lấn,
nên điểm không khớp tuyệt đối với evaluate_board, nhưng cùng chiều "ai đang hơn".
Ý tưởng cửa sổ trượt: tham khảo Allis (1994) và Mark Liu, AlphaGo Simplified.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from game.patterns import (
    PATTERN_SCORES,
    PATTERN_BLOCKED_FOUR,
    PATTERN_BLOCKED_THREE,
    PATTERN_BLOCKED_TWO,
    PATTERN_FIVE,
    PATTERN_OPEN_FOUR,
    PATTERN_OPEN_THREE,
    PATTERN_OPEN_TWO,
)

WALL = 2  # giá trị giả cho ô ngoài biên


def _build_scores() -> np.ndarray:
    """scores[số quân][số đầu mở]; với 4 quân, chỉ số = số điểm hoàn thành (1 hoặc 2)."""
    s = np.zeros((6, 3), dtype=np.float64)
    s[2] = [0, PATTERN_SCORES[PATTERN_BLOCKED_TWO], PATTERN_SCORES[PATTERN_OPEN_TWO]]
    blocked3 = PATTERN_SCORES[PATTERN_BLOCKED_THREE]
    s[3] = [blocked3, blocked3, PATTERN_SCORES[PATTERN_OPEN_THREE]]
    s[4] = [0, PATTERN_SCORES[PATTERN_BLOCKED_FOUR], PATTERN_SCORES[PATTERN_OPEN_FOUR]]
    s[5] = PATTERN_SCORES[PATTERN_FIVE]
    return s


SCORES = _build_scores()


@njit(cache=True)
def _eval_numba(board: np.ndarray, player: int, scores: np.ndarray) -> float:
    n = board.shape[0]
    total = 0.0
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for d in range(4):
        dr, dc = dirs[d]
        for r in range(n):
            for c in range(n):
                er = r + 4 * dr
                ec = c + 4 * dc
                if er < 0 or er >= n or ec < 0 or ec >= n:
                    continue
                mine = 0
                theirs = 0
                empty_idx = -1
                for k in range(5):
                    v = board[r + k * dr, c + k * dc]
                    if v == player:
                        mine += 1
                    elif v != 0:
                        theirs += 1
                    else:
                        empty_idx = k
                if mine > 0 and theirs == 0:
                    owner = player
                    cnt = mine
                    sign = 1.0
                elif theirs > 0 and mine == 0:
                    owner = -player
                    cnt = theirs
                    sign = -1.0
                else:
                    continue

                # Hai ô ngay ngoài cửa sổ (biên bàn = WALL, không tính là chặn).
                before = WALL
                br = r - dr
                bc = c - dc
                if 0 <= br < n and 0 <= bc < n:
                    before = board[br, bc]
                after = WALL
                ar = er + dr
                ac = ec + dc
                if 0 <= ar < n and 0 <= ac < n:
                    after = board[ar, ac]

                if cnt == 5:
                    # Luật VN: năm bị đối thủ chặn cả 2 đầu không phải thắng.
                    if before == -owner and after == -owner:
                        continue
                    total += sign * scores[5, 0]
                elif cnt == 4:
                    # Lấp đầy cửa sổ tạo thành năm bị chặn hai đầu thì không tính điểm.
                    if before == -owner and after == -owner:
                        continue
                    idx = 1
                    if (empty_idx == 0 and after == 0) or (empty_idx == 4 and before == 0):
                        idx = 2
                    total += sign * scores[4, idx]
                else:
                    idx = 0
                    if before == 0:
                        idx += 1
                    if after == 0:
                        idx += 1
                    total += sign * scores[cnt, idx]
    return total


def fast_evaluate(board: np.ndarray, player: int) -> float:
    """Điểm thế cờ từ góc nhìn của `player` (dương = player đang hơn)."""
    return float(_eval_numba(board, player, SCORES))
