"""Kiểm tra cấu hình thí nghiệm và tính tương đương minimax cho issue #8."""
import pytest
from analysis.ablation import benchmark_ab, experiment_configs
from arena.tournament import build_schedule
from arena.agent_factory import AgentSpec
from agents.alphabeta_agent import AlphaBetaAgent
from game.game import CaroGame


def test_experiment_matrix():
    configs = experiment_configs()
    sims, rollout, *rq1 = configs
    assert [sims['agents'][f'mcts_{n}']['params']['simulations']
            for n in (100, 400, 1600, 6400)] == [100, 400, 1600, 6400]
    assert sims['mode'] == rollout['mode'] == 'B'
    assert [c['budget'] for c in rq1] == [0.1, 0.5, 1.0]
    assert all(c['mode'] == 'A' for c in rq1)
    params = [a['params'] for a in rollout['agents'].values()]
    assert {p['rollout'] for p in params} == {'random', 'heuristic'}
    assert all(p['simulations'] == 1600 and not p['use_forced_move'] for p in params)


def test_puct_config_still_registered():
    assert AgentSpec.from_config('neural', {'type': 'puct'}).type == 'puct'


def test_smoke_rollout_is_explicitly_reduced():
    cfg = experiment_configs(smoke=True)[1]
    assert all(a['params']['simulations'] == 20 for a in cfg['agents'].values())
    assert cfg['name'].endswith('_smoke')


def test_rq1_pairs_swap_sides_and_reuse_openings_and_seeds():
    cfg = experiment_configs()[-1]
    openings = [{'id': i, 'moves': [60, 61]} for i in range(100)]
    tasks = build_schedule(cfg, openings)
    assert len(tasks) == 600
    a, b = tasks[:2]
    assert a.agent_x == b.agent_o and a.agent_o == b.agent_x
    assert a.opening_moves == b.opening_moves
    assert a.seed == b.seed
    assert {t.seed for t in tasks} == {0, 1, 2}


@pytest.mark.parametrize('depth', [2, 3])
def test_four_ab_variants_preserve_minimax_score(depth):
    rows = benchmark_ab([{'id': 0, 'moves': [60, 61, 72, 48]}], depth=depth)
    assert len(rows) == 4
    assert all(r['same_score'] for r in rows)
    baseline = rows[0]
    assert baseline['cutoffs'] == 0
    assert all(r['tt_hits'] == 0 for r in rows)
    pruned_unordered = next(r for r in rows if r['pruning'] and not r['ordering'])
    assert pruned_unordered['same_action']
    assert pruned_unordered['nodes'] <= baseline['nodes']


def test_ordering_off_ignores_tt_move(monkeypatch):
    game = CaroGame()
    game.step(60)
    agent = AlphaBetaAgent(depth=1, use_move_ordering=False)
    moves = agent._get_moves(game)
    key = agent._hash(game)
    agent._tt[key] = (0, 0, 0.0, moves[-1])
    visited = []

    def record(state, depth, alpha, beta, ply):
        visited.append(state.last_move)
        return 0.0

    monkeypatch.setattr(agent, '_negamax', record)
    agent._search_root(game, 1)
    assert visited == moves
