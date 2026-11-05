"""Hàm lượng giá nhanh (Numba) cho Alpha-Beta và rollout MCTS.

Quét mọi cửa sổ 5 ô theo 4 hướng; cửa sổ chỉ chứa quân một bên thì cộng điểm
theo số quân. Không cho đúng điểm như game.patterns.evaluate_board (cách đếm
khác), nhưng cùng thang điểm cỡ và cùng chiều "ai đang hơn".
Ý tưởng cửa sổ trượt: tham khảo Allis (1994) và Mark Liu, AlphaGo Simplified.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Điểm theo số quân trong một cửa sổ 5 ô không bị đối phương chiếm.
# Chỉ số = số quân (0..5). Cùng cỡ với PATTERN_SCORES trong game/patterns.py.
WINDOW_SCORES = np.array([0, 0, 50, 1_000, 100_000, 100_000_000], dtype=np.float64)
WIN_STREAK = 5


@njit(cache=True)
def _eval_numba(board: np.ndarray, player: int, scores: np.ndarray) -> float:
    n = board.shape[0]
    total = 0.0
    # 4 hướng: ngang, dọc, chéo xuống, chéo lên
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
                for k in range(5):
                    v = board[r + k * dr, c + k * dc]
                    if v == player:
                        mine += 1
                    elif v != 0:
                        theirs += 1
                if mine > 0 and theirs == 0:
                    total += scores[mine]
                elif theirs > 0 and mine == 0:
                    total -= scores[theirs]
    return total


def fast_evaluate(board: np.ndarray, player: int) -> float:
    """Điểm thế cờ từ góc nhìn của `player` (dương = player đang hơn)."""
    return float(_eval_numba(board, player, WINDOW_SCORES))