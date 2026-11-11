"""Validate the complete issue #8 run and export reviewable CSV results.

Implementation assisted by OpenAI Codex. Reuses the project's arena statistics.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import numba
import yaml

from analysis.ablation import experiment_configs
from analysis.arena_stats import pair_table, timing_table
from arena.openings import load_openings
from arena.tournament import build_schedule
from game.game import CaroGame


def validate_games(cfg, games):
    tasks = {t.game_id: t for t in build_schedule(cfg, load_openings(cfg['openings_file']))}
    ids = [g['game_id'] for g in games]
    if len(set(ids)) != len(ids) or set(ids) != set(tasks):
        raise ValueError(f"{cfg['name']}: missing, unexpected or duplicate games")
    for g in games:
        t = tasks[g['game_id']]
        if g.get('error') is not None:
            raise ValueError(f"Agent error: {g['game_id']}: {g['error']}")
        expected = dict(experiment=t.experiment, mode=t.mode, budget=t.budget,
                        size=t.size, rule=t.rule, opening_id=t.opening_id,
                        opening_moves=list(t.opening_moves), seed=t.seed,
                        agent_x={**t.agent_x.to_dict(), 'seed': t.seed_x},
                        agent_o={**t.agent_o.to_dict(), 'seed': t.seed_o})
        if any(g[k] != v for k, v in expected.items()):
            raise ValueError(f"Configuration mismatch: {g['game_id']}")
        game = CaroGame(size=t.size, rule=t.rule)
        for action in t.opening_moves:
            game.step(action)
        for move in g['moves']:
            if move['player'] != game.current_player or move['time_ms'] < 0:
                raise ValueError(f"Invalid turn/timing: {g['game_id']}")
            if move['time_ms'] > 600_000:
                raise ValueError(f"Suspended timing sample; rerun this game: {g['game_id']}")
            spec = t.agent_x if move['player'] == 1 else t.agent_o
            if t.mode == 'B' and spec.type == 'mcts':
                if move['stats']['sims'] != spec.params['simulations']:
                    raise ValueError(f"Simulation count mismatch: {g['game_id']}")
            game.step(move['action'])
        if not game.is_terminal() or game.result() != g['result']:
            raise ValueError(f"Result mismatch: {g['game_id']}")
        winner = t.agent_x.label if g['result'] == 1 else t.agent_o.label if g['result'] == -1 else None
        if g['winner'] != winner or g['num_moves'] != game.move_count:
            raise ValueError(f"Winner/move count mismatch: {g['game_id']}")


def cluster_interval(games, label):
    """Bootstrap complete opening blocks, keeping both colors and all seeds."""
    blocks = defaultdict(list)
    for g in games:
        won = (g['result'] == 1 and g['agent_x']['label'] == label) or (
            g['result'] == -1 and g['agent_o']['label'] == label)
        blocks[g['opening_id']].append(0.5 if g['result'] == 0 else float(won))
    means = np.array([np.mean(blocks[key]) for key in sorted(blocks)])
    rng = np.random.default_rng(2026)
    boot = means[rng.integers(0, len(means), size=(5000, len(means)))].mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_ab_rows(rows, opening_ids):
    """Yêu cầu đủ bốn biến thể ở sâu 2 và 3 trên cùng các khai cuộc."""
    expected = {(str(oid), str(depth), str(pruning), str(ordering))
                for oid in opening_ids for depth in (2, 3)
                for pruning in (False, True) for ordering in (False, True)}
    keys = [(r['opening_id'], r['depth'], r['pruning'], r['ordering']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Thiếu hoặc trùng biến thể Alpha-Beta ở độ sâu 2 và 3')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['opening_id'], row['depth'])].append(row)
        if row['same_score'] != 'True' or int(row['tt_hits']) != 0:
            raise ValueError('Điểm minimax khác nhau hoặc TT chưa được tắt')
    for group in groups.values():
        if len({float(r['score']) for r in group}) != 1:
            raise ValueError('Bốn biến thể phải giữ nguyên điểm minimax')
        reference = next(r for r in group if r['pruning'] == r['ordering'] == 'False')
        if any((r['same_action'] == 'True') != (r['action'] == reference['action']) for r in group):
            raise ValueError('Cột cùng nước không khớp nước đi thực tế')


def export_results(source: Path, dest: Path):
    pairs, timing, inventory, configs = [], [], [], experiment_configs()
    archives = {}
    for cfg in configs:
        name = cfg['name']
        if yaml.safe_load((source / f'{name}.yaml').read_text(encoding='utf-8')) != cfg:
            raise ValueError(f'Configuration mismatch: {name}')
        raw = (source / f'{name}.jsonl').read_bytes()
        games = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        validate_games(cfg, games)
        archives[f'{name}.jsonl.gz'] = gzip.compress(raw, mtime=0)
        inventory.append(dict(experiment=name, games=len(games), errors=0,
                              log_sha256=hashlib.sha256(raw).hexdigest()))
        table = pair_table(games)
        labels = list(cfg['agents'])
        comparisons = cfg.get('pairs') or [[labels[0], labels[1]]]
        for a, b in comparisons:
            subset = [g for g in games if {g['agent_x']['label'], g['agent_o']['label']} == {a, b}]
            row = table[(a, b)]
            lo, hi = cluster_interval(subset, a)
            pairs.append(dict(experiment=name, agent_a=a, agent_b=b, games=row['n'],
                              wins_a=row['wins'], draws=row['draws'], losses_a=row['losses'],
                              score_a=row['score'], wilson_low=row['ci95'][0], wilson_high=row['ci95'][1],
                              opening_bootstrap_low=lo, opening_bootstrap_high=hi))
        for label, row in timing_table(games).items():
            moves = [m for g in games for m in g['moves']
                     if g['agent_x' if m['player'] == 1 else 'agent_o']['label'] == label]
            late = sum(m['time_ms'] > cfg['budget'] * 1100 for m in moves) if cfg['budget'] else None
            timing.append(dict(experiment=name, agent=label, budget_seconds=cfg['budget'],
                               **row, over_budget_10pct=late))
    with (source / 'alphabeta.csv').open(encoding='utf-8') as stream:
        ab_rows = list(csv.DictReader(stream))
    validate_ab_rows(ab_rows, [o['id'] for o in load_openings('configs/openings.json')[:10]])
    repair_path = source / 'suspension_repair.json'
    repair = json.loads(repair_path.read_text(encoding='utf-8')) if repair_path.exists() else None
    if repair:
        rollout_hash = next(r['log_sha256'] for r in inventory if r['experiment'] == 'ablation_rollout')
        if repair['repaired_log_sha256'] != rollout_hash:
            raise ValueError('Timing-repair manifest does not match the validated rollout log')
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / 'pairs.csv', pairs)
    write_csv(dest / 'timing.csv', timing)
    write_csv(dest / 'validation.csv', inventory)
    write_csv(dest / 'alphabeta.csv', ab_rows)
    for name, data in archives.items():
        (dest / name).write_bytes(data)
    snapshot_path = source / 'run_source_sha256.json'
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ('agents', 'arena', 'game', 'analysis')
                     for p in sorted(Path(folder).glob('*.py'))}
    metadata = dict(configs=configs, software={'numpy': np.__version__, 'numba': numba.__version__},
                    source_revision_note='Agents ran from the PR working tree. Git revisions in run_metadata are base revisions; source_sha256 records the source included with these results.',
                    environment=json.loads((source / 'environment.json').read_text(encoding='utf-8')),
                    run_metadata={c['name']: [json.loads(line) for line in
                        (source / f"{c['name']}.meta.jsonl").read_text(encoding='utf-8').splitlines() if line.strip()]
                        for c in configs},
                    notes=['5000 bootstrap samples by opening; both colors and three seeds stay together.',
                           'All games replayed and checked against the exact planned schedule.',
                           'Time budgets are soft; over_budget_10pct counts moves exceeding budget by 10%.',
                           'Worker counts are recorded per run; timings depend on shared CPU load.'],
                    source_sha256=json.loads(snapshot_path.read_text(encoding='utf-8')) if snapshot_path.exists() else source_hashes,
                    validation_source_sha256=source_hashes,
                    openings_sha256=hashlib.sha256(Path('configs/openings.json').read_bytes()).hexdigest())
    if repair:
        metadata['timing_repair'] = repair
        metadata['notes'].append('excluded_suspension.jsonl.gz contains audit-only records, excluded from all results.')
        (dest / 'excluded_suspension.jsonl.gz').write_bytes((source / 'excluded_suspension.jsonl.gz').read_bytes())
    ab_measurement = source / 'ab_measurement.json'
    if ab_measurement.exists():
        metadata['alphabeta_measurement'] = json.loads(ab_measurement.read_text(encoding='utf-8'))
    (dest / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Validated {sum(r["games"] for r in inventory)} games -> {dest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/issue8'))
    parser.add_argument('--wait', action='store_true', help='Wait for the runner to finish its report')
    args = parser.parse_args()
    if args.wait:
        deadline = time.monotonic() + 24 * 3600
        while not (args.source / 'report.md').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Runner did not finish within 24 hours')
            time.sleep(30)
    export_results(args.source, args.out)
