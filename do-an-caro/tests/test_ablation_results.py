"""Reject incomplete or corrupt data; OpenAI Codex assisted tests for issue #8."""
import copy
import json

import pytest

from analysis.ablation_results import validate_games
from analysis.ablation_results import validate_ab_rows
from arena.match import play_game
from arena.tournament import build_schedule


@pytest.fixture
def completed(tmp_path):
    openings = [{'id': 0, 'moves': []}]
    path = tmp_path / 'openings.json'
    path.write_text(json.dumps({'openings': openings}), encoding='utf-8')
    cfg = dict(name='test', mode='B', budget=None, size=5, rule='vietnam',
               openings_file=str(path), num_openings=1, seeds=[0],
               agents={'a': {'type': 'random'}, 'b': {'type': 'random'}})
    return cfg, [play_game(t) for t in build_schedule(cfg, openings)]


def test_accepts_complete_replayable_schedule(completed):
    validate_games(*completed)


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'error', 'suspension', 'turn', 'result', 'params'])
def test_rejects_bad_run(completed, corruption):
    cfg, games = copy.deepcopy(completed)
    if corruption == 'missing':
        games.pop()
    elif corruption == 'duplicate':
        games.append(games[0])
    elif corruption == 'error':
        games[0]['error'] = 'RuntimeError'
    elif corruption == 'suspension':
        games[0]['moves'][0]['time_ms'] = 600_001
    elif corruption == 'turn':
        games[0]['moves'][0]['player'] *= -1
    elif corruption == 'result':
        games[0]['result'] = 99
    else:
        games[0]['agent_x']['params'] = {'changed': True}
    with pytest.raises(ValueError):
        validate_games(cfg, games)


def test_ab_results_require_both_depths_and_unique_variants():
    rows = [dict(opening_id='0', depth=str(depth), pruning=str(pruning),
                 ordering=str(ordering), same_score='True', same_action='True',
                 tt_hits='0', score='100', action='60')
            for depth in (2, 3) for pruning in (False, True) for ordering in (False, True)]
    validate_ab_rows(rows, [0])
    with pytest.raises(ValueError):
        validate_ab_rows(rows[:4], [0])
    with pytest.raises(ValueError):
        validate_ab_rows(rows[:-1] + [rows[0]], [0])
    rows[-1]['score'] = '200'
    with pytest.raises(ValueError):
        validate_ab_rows(rows, [0])
