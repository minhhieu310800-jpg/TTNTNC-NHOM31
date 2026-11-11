"""Compile the existing immediate-win scan without changing its move order.

Implementation assisted by OpenAI Codex. Win rules reuse game.rules.check_win_at.
"""
from numba import njit

from game.rules import check_win_at

_check_win = njit(cache=True)(check_win_at)


@njit(cache=True)
def local_rollout_moves(board, last_move, radius=2):
    """Same row-major local candidates and full-board fallback as MCTSAgent."""
    size = board.shape[0]
    if last_move == -1:
        return [size // 2 * size + size // 2]
    r0, c0 = last_move // size, last_move % size
    moves = []
    for r in range(max(0, r0 - radius), min(size, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(size, c0 + radius + 1)):
            if board[r, c] == 0:
                moves.append(r * size + c)
    if not moves:
        for r in range(size):
            for c in range(size):
                if board[r, c] == 0:
                    moves.append(r * size + c)
    return moves


@njit(cache=True)
def advance_rollout(board, action, player, rule, win_streak):
    """Apply one legal rollout move to a private board and check the same rules."""
    r, c = action // board.shape[1], action % board.shape[1]
    board[r, c] = player
    return _check_win(board, r, c, rule, win_streak)


@njit(cache=True)
def tactical_rollout_action(board, moves, player, rule):
    """First winning move, then first block, or -1; leave the board unchanged."""
    size = board.shape[1]
    for side in (player, -player):
        for action in moves:
            r, c = action // size, action % size
            if board[r, c] != 0:
                continue
            board[r, c] = side
            won = _check_win(board, r, c, rule, 5)
            board[r, c] = 0
            if won:
                return action
    return -1
