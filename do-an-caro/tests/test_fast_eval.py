import random

import numpy as np

from agents.fast_eval import fast_evaluate
from game.game import CaroGame
from game.patterns import evaluate_board


def board_with(stones):
    board = np.zeros((11, 11), dtype=np.int8)
    for (r, c), v in stones.items():
        board[r, c] = v
    return board


def row(cols, v, r=5):
    return {(r, c): v for c in cols}


def test_empty_board_is_zero():
    assert fast_evaluate(CaroGame().board, 1) == 0.0


def test_sign_flips_with_player():
    game = CaroGame()
    for coord in [(5, 5), (0, 0), (5, 6), (0, 2), (5, 7)]:
        game.step(coord)
    assert fast_evaluate(game.board, 1) == -fast_evaluate(game.board, -1)
    assert fast_evaluate(game.board, 1) > 0


def test_sign_agrees_with_evaluate_board():
    agree = total = 0
    for seed in range(30):
        rng = random.Random(seed)
        game = CaroGame()
        for _ in range(14):
            game.step(rng.choice(game.legal_actions()))
            if game.is_terminal():
                break
        old = evaluate_board(game.board, 1, game.rule)
        new = fast_evaluate(game.board, 1)
        if old != 0 and new != 0:
            total += 1
            agree += (old > 0) == (new > 0)
    assert total > 0
    assert agree / total >= 0.7, f"ti le cung dau: {agree}/{total}"


def test_five_blocked_both_ends_is_not_a_win():
    blocked = board_with({**row(range(2, 7), 1), (5, 1): -1, (5, 7): -1})
    free = board_with({**row(range(2, 7), 1), (5, 1): -1})
    assert fast_evaluate(blocked, 1) < 100_000_000
    assert fast_evaluate(free, 1) >= 100_000_000


def test_open_four_beats_blocked_four():
    open_four = board_with(row(range(3, 7), 1))
    blocked_four = board_with({**row(range(3, 7), 1), (5, 2): -1, (5, 7): -1})
    assert fast_evaluate(open_four, 1) > fast_evaluate(blocked_four, 1)


def test_open_three_beats_blocked_three():
    open_three = board_with(row(range(4, 7), 1))
    blocked_three = board_with({**row(range(4, 7), 1), (5, 3): -1, (5, 7): -1})
    assert fast_evaluate(open_three, 1) > fast_evaluate(blocked_three, 1)
