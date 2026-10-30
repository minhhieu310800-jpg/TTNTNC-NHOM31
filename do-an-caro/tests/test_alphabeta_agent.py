from agents.alphabeta_agent import AlphaBetaAgent
from game.game import CaroGame


def play(game: CaroGame, coords: list[tuple[int, int]]) -> CaroGame:
    """Đánh lần lượt các toạ độ (X đi trước, xen kẽ với O)."""
    for coord in coords:
        game.step(coord)
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