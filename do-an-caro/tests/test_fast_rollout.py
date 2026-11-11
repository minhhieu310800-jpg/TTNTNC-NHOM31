"""Preserve the original heuristic, board and RNG; OpenAI Codex assisted tests."""
import random

import numpy as np
import pytest

from agents.fast_rollout import local_rollout_moves, tactical_rollout_action
from agents.mcts_agent import MCTSAgent
from game.game import CaroGame
from game.patterns import is_immediate_win


def reference(game, moves):
    for side in (game.current_player, -game.current_player):
        for action in moves:
            r, c = game.action_to_coord(action)
            if is_immediate_win(game.board, r, c, side, rule=game.rule):
                return action
    return -1


@pytest.mark.parametrize('rule', ['vietnam', 'freestyle'])
def test_scan_matches_original_on_seeded_games(rule):
    rng = random.Random(2026)
    agent = MCTSAgent(rollout='heuristic', seed=8)
    for _ in range(12):
        game = CaroGame(rule=rule)
        while not game.is_terminal():
            moves = agent._local_moves(game)
            assert local_rollout_moves(game.board, -1 if game.last_move is None else game.last_move, 2) == moves
            board = game.board.copy()
            actual = tactical_rollout_action(
                game.board, np.asarray(moves, dtype=np.int64), game.current_player, rule,
            )
            assert actual == reference(game, moves)
            np.testing.assert_array_equal(game.board, board)
            game.step(rng.choice(game.legal_actions()))


@pytest.mark.parametrize('rollout', ['random', 'heuristic'])
@pytest.mark.parametrize('seed', [0, 7, 19])
def test_full_search_preserves_rng_and_tree_statistics(monkeypatch, rollout, seed):
    game = CaroGame()
    for action in [60, 61, 72, 48]:
        game.step(action)
    fast = MCTSAgent(simulations=100, rollout=rollout, seed=seed)
    slow = MCTSAgent(simulations=100, rollout=rollout, seed=seed)

    def original_action(state):
        moves = slow._local_moves(state)
        action = reference(state, moves) if rollout == 'heuristic' else -1
        return action if action != -1 else slow.rng.choice(moves)

    def original_result(state):
        steps = 0
        try:
            while not state.is_terminal():
                state.step(original_action(state))
                steps += 1
            return state.result()
        finally:
            for _ in range(steps):
                state.undo()

    monkeypatch.setattr(slow, '_rollout_action', original_action)
    monkeypatch.setattr(slow, '_rollout_result', original_result)
    assert fast.select_action(game) == slow.select_action(game)
    for key in ('sims', 'best_visits', 'win_rate'):
        assert fast.last_stats[key] == slow.last_stats[key]
    assert fast.rng.getstate() == slow.rng.getstate()
