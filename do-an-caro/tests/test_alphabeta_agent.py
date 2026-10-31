import random

from agents.alphabeta_agent import AlphaBetaAgent
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
    action = AlphaBetaAgent().select_action(game)
    assert game.action_to_coord(action) in [(5, 2), (5, 7)]


def test_blocks_opponent_win():
    # O có 4 quân hàng 5 (cột 3-6), đầu trái bị X chặn ở (5,2); X phải chặn (5,7).
    game = play(CaroGame(), [
        (5, 2), (5, 3), (0, 2), (5, 4), (0, 4), (5, 5), (10, 10), (5, 6),
    ])
    action = AlphaBetaAgent().select_action(game)
    assert game.action_to_coord(action) == (5, 7)


def test_state_unchanged_after_search():
    game = play(CaroGame(), [(5, 5), (5, 6), (6, 6), (4, 4)])
    board_before = game.board.copy()
    player_before = game.current_player
    moves_before = game.move_count

    AlphaBetaAgent().select_action(game)

    assert (game.board == board_before).all()
    assert game.current_player == player_before
    assert game.move_count == moves_before


def test_pruning_same_move_fewer_nodes():
    for seed in range(3):
        game = random_position(seed, n_moves=8)
        on = AlphaBetaAgent(depth=2, use_pruning=True, use_move_ordering=False)
        off = AlphaBetaAgent(depth=2, use_pruning=False, use_move_ordering=False)
        assert on.select_action(game) == off.select_action(game)
        assert on.last_stats["score"] == off.last_stats["score"]
        assert on.last_stats["nodes"] <= off.last_stats["nodes"]


def test_pruning_has_cutoffs():
    game = random_position(0, n_moves=8)
    agent = AlphaBetaAgent(depth=2, use_pruning=True)
    agent.select_action(game)
    assert agent.last_stats["cutoffs"] > 0

def test_move_ordering_same_score():
    for seed in range(2):
        game = random_position(seed, n_moves=8)
        ordered = AlphaBetaAgent(depth=2, use_move_ordering=True)
        plain = AlphaBetaAgent(depth=2, use_move_ordering=False)
        ordered.select_action(game)
        plain.select_action(game)
        # Chỉ so score: nhiều nước bằng điểm thì nước được chọn có thể khác.
        assert ordered.last_stats["score"] == plain.last_stats["score"]

def test_budget_returns_legal_move_in_time():
    game = random_position(1, n_moves=8)
    board_before = game.board.copy()
    agent = AlphaBetaAgent()
    action = agent.select_action(game, budget=0.3)
    assert action in game.legal_actions()
    assert agent.last_stats["time_used"] < 1.5
    assert (game.board == board_before).all()


def test_tiny_budget_still_returns_legal_move():
    game = random_position(2, n_moves=8)
    action = AlphaBetaAgent().select_action(game, budget=0.001)
    assert action in game.legal_actions()


def test_no_budget_uses_fixed_depth():
    game = random_position(3, n_moves=8)
    agent = AlphaBetaAgent(depth=2)
    agent.select_action(game)
    assert agent.last_stats["depth_reached"] == 2