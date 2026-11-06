"""Kiểm thử thống kê arena: Wilson, ma trận đối đầu, màu quân, thời gian, Elo."""

from __future__ import annotations

import pytest

from analysis.arena_stats import (
    bootstrap_elo,
    bradley_terry,
    color_table,
    pair_table,
    timing_table,
    wilson_interval,
)
from analysis.summarize import summarize


def _game(x: str, o: str, result: int, moves=None) -> dict:
    return {"agent_x": {"label": x}, "agent_o": {"label": o}, "result": result, "moves": moves or []}


def _series(a: str, b: str, a_wins: int, draws: int, b_wins: int) -> list[dict]:
    """Chuỗi ván a gặp b, chia đều màu quân."""
    games = []
    outcomes = ["a"] * a_wins + ["d"] * draws + ["b"] * b_wins
    for i, out in enumerate(outcomes):
        x, o = (a, b) if i % 2 == 0 else (b, a)
        if out == "d":
            res = 0
        else:
            winner = a if out == "a" else b
            res = 1 if winner == x else -1
        games.append(_game(x, o, res))
    return games


def test_wilson_matches_known_values():
    lo, hi = wilson_interval(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)
    lo, hi = wilson_interval(0.0, 12)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2425, abs=1e-4)


def test_pair_table_score_and_symmetry():
    games = _series("A", "B", a_wins=6, draws=2, b_wins=2)
    t = pair_table(games)
    ab, ba = t[("A", "B")], t[("B", "A")]
    assert (ab["n"], ab["wins"], ab["draws"], ab["losses"]) == (10, 6, 2, 2)
    assert ab["score"] == pytest.approx(0.7)
    assert ab["score"] + ba["score"] == pytest.approx(1.0)
    assert ab["ci95"][0] < 0.7 < ab["ci95"][1]


def test_color_table_detects_first_player_advantage():
    games = [_game("A", "B", 1), _game("B", "A", 1), _game("A", "B", 1), _game("B", "A", 0)]
    ct = color_table(games)
    assert ct["x_score"] == pytest.approx((1 + 1 + 1 + 0.5) / 4)
    assert ct["agents"]["A"]["score_as_x"] == 1.0
    assert ct["agents"]["A"]["score_as_o"] == pytest.approx(0.25)


def test_timing_table_mean_p95_and_costs():
    moves = [{"player": 1, "time_ms": t, "stats": {"nodes": 10}} for t in range(1, 101)]
    moves += [{"player": -1, "time_ms": 5.0, "stats": {"sims": 200}}]
    tt = timing_table([_game("AB", "M", 1, moves)])
    assert tt["AB"]["mean_ms"] == pytest.approx(50.5)
    assert tt["AB"]["p95_ms"] == pytest.approx(95.05)
    assert tt["AB"]["mean_nodes"] == 10 and tt["AB"]["mean_sims"] is None
    assert tt["M"]["mean_sims"] == 200


def test_bradley_terry_orders_agents_and_is_centered():
    games = (_series("A", "B", 16, 0, 4) + _series("B", "C", 16, 0, 4) + _series("A", "C", 19, 0, 1))
    elo = bradley_terry(games)
    assert elo["A"] > elo["B"] > elo["C"]
    assert sum(elo.values()) == pytest.approx(0.0, abs=1e-6)
    # thắng 80% ~ chênh khoảng 240 Elo; prior kéo nhẹ về 0
    assert 150 < elo["A"] - elo["B"] < 300


def test_bradley_terry_finite_when_one_agent_never_wins():
    elo = bradley_terry(_series("A", "B", 10, 0, 0))
    assert all(abs(v) < 2000 for v in elo.values())


def test_bootstrap_ci_contains_point_estimate():
    games = _series("A", "B", 14, 2, 4) + _series("B", "C", 12, 2, 6)
    res = bootstrap_elo(games, n_boot=200, seed=1)
    for v in res.values():
        lo, hi = v["ci95"]
        assert lo <= v["elo"] <= hi


def test_summarize_produces_all_sections(tmp_path):
    import json
    path = tmp_path / "log.jsonl"
    games = _series("A", "B", 6, 2, 2)
    path.write_text("\n".join(json.dumps(g) for g in games) + "\n", encoding="utf-8")
    text = summarize(str(path), n_boot=50)
    for heading in ("Ma trận đối đầu", "Kết quả từng cặp", "Lợi thế đi trước", "Thời gian", "Xếp hạng"):
        assert heading in text
