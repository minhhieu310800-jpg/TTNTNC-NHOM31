import random

from agents.fast_eval import fast_evaluate
from game.game import CaroGame
from game.patterns import evaluate_board


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
    for seed in range(20):
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
    assert total > 0 and agree / total >= 0.7