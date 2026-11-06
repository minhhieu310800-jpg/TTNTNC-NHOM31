from agents.mcts_agent import MCTSAgent
from game.game import CaroGame


def play(game, coords):
    for coord in coords:
        game.step(coord)
    return game


OPENING = [(5, 5), (5, 6), (6, 6), (4, 4)]
WIN_IN_ONE = [
    (5, 3), (0, 0), (5, 4), (0, 2), (5, 5), (0, 4), (5, 6), (10, 10),
]


def test_takes_immediate_win():
    game = play(CaroGame(), WIN_IN_ONE)
    action = MCTSAgent(simulations=20, seed=0, use_forced_move=True).select_action(game)
    assert game.action_to_coord(action) in [(5, 2), (5, 7)]


def test_blocks_opponent_win():
    game = play(CaroGame(), [
        (5, 2), (5, 3), (0, 2), (5, 4), (0, 4), (5, 5), (10, 10), (5, 6),
    ])
    action = MCTSAgent(simulations=20, seed=0, use_forced_move=True).select_action(game)
    assert game.action_to_coord(action) == (5, 7)


def test_state_unchanged_after_search():
    game = play(CaroGame(), OPENING)
    board_before = game.board.copy()
    player_before = game.current_player
    moves_before = game.move_count
    MCTSAgent(simulations=50, rollout="random", seed=0).select_action(game)
    assert (game.board == board_before).all()
    assert game.current_player == player_before
    assert game.move_count == moves_before


def test_same_seed_same_move():
    game = play(CaroGame(), OPENING)
    a = MCTSAgent(simulations=50, rollout="random", seed=7).select_action(game)
    b = MCTSAgent(simulations=50, rollout="random", seed=7).select_action(game)
    assert a == b


def test_tree_search_runs_without_forced_move():
    game = play(CaroGame(), OPENING)
    agent = MCTSAgent(simulations=100, rollout="random", seed=0)
    action = agent.select_action(game)
    assert action in game.legal_actions()
    assert agent.last_stats["sims"] == 100
    assert agent.last_stats["best_visits"] > 1
    assert 0.0 <= agent.last_stats["win_rate"] <= 1.0


def test_heuristic_rollout_runs():
    game = play(CaroGame(), OPENING)
    agent = MCTSAgent(simulations=30, rollout="heuristic", seed=0)
    assert agent.select_action(game) in game.legal_actions()


def test_budget_returns_legal_move_in_time():
    game = play(CaroGame(), OPENING)
    agent = MCTSAgent(seed=0)
    action = agent.select_action(game, budget=0.3)
    assert action in game.legal_actions()
    assert agent.last_stats["time_used"] < 1.5


def test_defaults_are_pure_mcts():
    agent = MCTSAgent()
    assert agent.name == "mcts"
    assert agent.rollout == "random"
    assert agent.use_forced_move is False
    assert agent.simulations == 1600


def test_pure_mcts_finds_win_with_enough_sims():
    # Kiểm chứng thực nghiệm: nếu fail, xóa test và ghi vào báo cáo RQ1
    # (MCTS thuần bỏ sót đòn thắng ngay), không tăng sims để cho qua.
    game = play(CaroGame(), WIN_IN_ONE)
    action = MCTSAgent(simulations=800, seed=0).select_action(game)
    assert game.action_to_coord(action) in [(5, 2), (5, 7)]


def test_win_rate_in_unit_interval():
    game = play(CaroGame(), OPENING)
    agent = MCTSAgent(simulations=50, seed=0)
    agent.select_action(game)
    assert 0.0 <= agent.last_stats["win_rate"] <= 1.0


def test_reset_seed_reproducible():
    game = play(CaroGame(), OPENING)
    a = MCTSAgent(simulations=50, seed=1)
    b = MCTSAgent(simulations=50, seed=2)
    a.reset_seed(5)
    b.reset_seed(5)
    assert a.select_action(game) == b.select_action(game)


def test_terminal_state_does_not_crash():
    game = play(CaroGame(), WIN_IN_ONE + [(5, 7)])
    assert game.is_terminal()
    assert MCTSAgent(simulations=10, seed=0).select_action(game) == -1