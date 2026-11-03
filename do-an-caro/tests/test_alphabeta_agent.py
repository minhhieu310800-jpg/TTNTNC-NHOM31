import random

from agents.alphabeta_agent import (
    MATE_SCORE_THRESHOLD,
    WIN_SCORE,
    AlphaBetaAgent,
)
from game.game import CaroGame


def play(game: CaroGame, coords: list[tuple[int, int]]) -> CaroGame:
    """Đánh lần lượt các toạ độ (X đi trước, xen kẽ với O)."""
    for coord in coords:
        game.step(coord)
    return game


def random_position(seed: int, n_moves: int) -> CaroGame:
    rng = random.Random(seed)
    game = CaroGame()
    for _ in range(n_moves):
        game.step(rng.choice(game.legal_actions()))
    return game


def test_takes_immediate_win():
    # X có 4 quân hàng 5 (cột 3-6), O rải xa; đến lượt X.
    game = play(CaroGame(), [
        (5, 3), (0, 0), (5, 4), (0, 2), (5, 5), (0, 4), (5, 6), (10, 10),
    ])
    action = AlphaBetaAgent(fast_eval=True).select_action(game)
    assert game.action_to_coord(action) in [(5, 2), (5, 7)]


def test_blocks_opponent_win():
    # O có 4 quân hàng 5 (cột 3-6), đầu trái bị X chặn ở (5,2); X phải chặn (5,7).
    game = play(CaroGame(), [
        (5, 2), (5, 3), (0, 2), (5, 4), (0, 4), (5, 5), (10, 10), (5, 6),
    ])
    action = AlphaBetaAgent(fast_eval=True).select_action(game)
    assert game.action_to_coord(action) == (5, 7)


def test_state_unchanged_after_search():
    game = play(CaroGame(), [(5, 5), (5, 6), (6, 6), (4, 4)])
    board_before = game.board.copy()
    player_before = game.current_player
    moves_before = game.move_count

    AlphaBetaAgent(fast_eval=True).select_action(game)

    assert (game.board == board_before).all()
    assert game.current_player == player_before
    assert game.move_count == moves_before


def test_pruning_same_move_fewer_nodes():
    for seed in range(3):
        game = random_position(seed, n_moves=8)
        on = AlphaBetaAgent(depth=2, use_pruning=True, use_move_ordering=False, fast_eval=True)
        off = AlphaBetaAgent(depth=2, use_pruning=False, use_move_ordering=False, fast_eval=True)
        assert on.select_action(game) == off.select_action(game)
        assert on.last_stats["score"] == off.last_stats["score"]
        assert on.last_stats["nodes"] <= off.last_stats["nodes"]


def test_pruning_has_cutoffs():
    game = random_position(0, n_moves=8)
    agent = AlphaBetaAgent(depth=2, use_pruning=True, fast_eval=True)
    agent.select_action(game)
    assert agent.last_stats["cutoffs"] > 0


def test_move_ordering_same_score():
    for seed in range(2):
        game = random_position(seed, n_moves=8)
        ordered = AlphaBetaAgent(depth=2, use_move_ordering=True, fast_eval=True)
        plain = AlphaBetaAgent(depth=2, use_move_ordering=False, fast_eval=True)
        ordered.select_action(game)
        plain.select_action(game)
        # Chỉ so score: nhiều nước bằng điểm thì nước được chọn có thể khác.
        assert ordered.last_stats["score"] == plain.last_stats["score"]


def test_budget_returns_legal_move_in_time():
    game = random_position(1, n_moves=8)
    board_before = game.board.copy()
    agent = AlphaBetaAgent(fast_eval=True)
    action = agent.select_action(game, budget=0.3)
    assert action in game.legal_actions()
    assert agent.last_stats["time_used"] < 1.5
    assert (game.board == board_before).all()


def test_tiny_budget_still_returns_legal_move():
    game = random_position(2, n_moves=8)
    action = AlphaBetaAgent(fast_eval=True).select_action(game, budget=0.001)
    assert action in game.legal_actions()


def test_no_budget_uses_fixed_depth():
    game = random_position(3, n_moves=8)
    agent = AlphaBetaAgent(depth=2, fast_eval=True)
    agent.select_action(game)
    assert agent.last_stats["depth_reached"] == 2


def test_cheap_ordering_same_score():
    game = random_position(1, n_moves=8)
    cheap = AlphaBetaAgent(depth=2, ordering="cheap", fast_eval=True)
    heur = AlphaBetaAgent(depth=2, ordering="heuristic", fast_eval=True)
    cheap.select_action(game)
    heur.select_action(game)
    assert cheap.last_stats["score"] == heur.last_stats["score"]


def test_default_depth_is_three():
    game = random_position(3, n_moves=8)
    agent = AlphaBetaAgent(fast_eval=True)
    agent.select_action(game)
    assert agent.last_stats["depth_reached"] == 3


def test_tt_same_score_and_has_hits():
    game = random_position(0, n_moves=8)
    on = AlphaBetaAgent(depth=4, use_tt=True, fast_eval=True)
    off = AlphaBetaAgent(depth=4, use_tt=False, fast_eval=True)
    on.select_action(game)
    off.select_action(game)
    assert on.last_stats["score"] == off.last_stats["score"]
    assert on.last_stats["nodes"] <= off.last_stats["nodes"]
    assert on.last_stats["tt_hits"] > 0


def test_transposition_move_is_tried_first():
    assert AlphaBetaAgent._prioritize_move([1, 2, 3], 2) == [2, 1, 3]
    assert AlphaBetaAgent._prioritize_move([1, 2, 3], 9) == [1, 2, 3]


def test_mate_scores_are_normalized_for_transposition_table():
    winning_score = WIN_SCORE - 7
    losing_score = -WIN_SCORE + 7

    stored_win = AlphaBetaAgent._score_to_tt(winning_score, ply=3)
    stored_loss = AlphaBetaAgent._score_to_tt(losing_score, ply=3)

    assert stored_win == WIN_SCORE - 4
    assert stored_loss == -WIN_SCORE + 4
    assert AlphaBetaAgent._score_from_tt(stored_win, ply=8) == WIN_SCORE - 12
    assert AlphaBetaAgent._score_from_tt(stored_loss, ply=8) == -WIN_SCORE + 12
    score = MATE_SCORE_THRESHOLD - 1
    assert AlphaBetaAgent._score_to_tt(score, ply=3) == score


def test_pattern_evaluator_is_default():
    assert AlphaBetaAgent().fast_eval is False
