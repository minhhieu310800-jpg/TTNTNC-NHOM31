"""Kiểm thử arena: logger, khai cuộc, lịch thi đấu, chơi ván và chạy tiếp (issue #12)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.base import Agent
from arena import agent_factory
from arena.agent_factory import AgentSpec, build_agent
from arena.logger import JsonlLogger, completed_game_ids, load_records, to_jsonable
from arena.match import GameTask, play_game
from arena.openings import canonical_key, generate_openings, load_openings, save_openings
from arena.tournament import build_schedule, load_config, run_tournament
from game.game import CaroGame

ROOT = Path(__file__).resolve().parent.parent
OPENINGS_FILE = ROOT / "configs" / "openings.json"
REQUIRED_FIELDS = {
    "game_id", "engine_version", "rule", "size", "opening_id", "seed",
    "agent_x", "agent_o", "moves", "result",
}


def _cfg(tmp_path: Path, agents: dict, seeds=(0, 1, 2), num_openings: int = 2, mode: str = "A") -> dict:
    openings = {"meta": {}, "openings": [
        {"id": 0, "moves": [60, 61]},
        {"id": 1, "moves": [48, 72, 50]},
        {"id": 2, "moves": [59, 49, 71, 61]},
    ]}
    path = tmp_path / "openings.json"
    path.write_text(json.dumps(openings), encoding="utf-8")
    return {
        "name": "test", "mode": mode, "budget": 0.01 if mode == "A" else None,
        "openings_file": str(path), "num_openings": num_openings,
        "seeds": list(seeds), "workers": 1, "agents": agents,
    }


# ---------------------------------------------------------------- logger

def test_to_jsonable_handles_numpy_and_inf():
    data = {"a": np.int64(3), "b": np.float32(0.5), "c": float("inf"), "d": (1, 2), "e": np.bool_(True)}
    assert to_jsonable(data) == {"a": 3, "b": 0.5, "c": None, "d": [1, 2], "e": True}
    json.dumps(to_jsonable(data))


def test_logger_roundtrip_and_skips_truncated_line(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = JsonlLogger(path)
    logger.write({"game_id": "g1", "score": float("nan")})
    logger.write({"game_id": "g2"})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"game_id": "g3", "moves": [1, 2')  # dòng cuối bị cắt khi tiến trình dừng
    records = load_records(path)
    assert [r["game_id"] for r in records] == ["g1", "g2"]
    assert records[0]["score"] is None
    assert completed_game_ids(path) == {"g1", "g2"}


# ---------------------------------------------------------------- openings

def test_canonical_key_same_for_symmetric_boards():
    board = np.zeros((11, 11), dtype=np.int8)
    board[5, 6] = 1
    board[4, 5] = -1
    for k in range(4):
        assert canonical_key(np.rot90(board, k)) == canonical_key(board)
        assert canonical_key(np.fliplr(np.rot90(board, k))) == canonical_key(board)


def test_generate_openings_reproducible_and_balanced(tmp_path):
    a = generate_openings(n=5, seed=7)
    b = generate_openings(n=5, seed=7)
    assert a["openings"] == b["openings"]
    keys = set()
    for op in a["openings"]:
        assert 2 <= len(op["moves"]) <= 4
        assert abs(op["eval"]) <= a["meta"]["balance_threshold"]
        game = CaroGame()
        for m in op["moves"]:
            r, c = divmod(m, 11)
            assert abs(r - 5) <= 2 and abs(c - 5) <= 2
            game.step(m)
        keys.add(canonical_key(game.board))
    assert len(keys) == 5  # không trùng qua đối xứng
    save_openings(a, tmp_path / "o.json")
    assert load_openings(tmp_path / "o.json") == a["openings"]


def test_committed_openings_file_has_100_valid_openings():
    openings = load_openings(OPENINGS_FILE)
    assert [op["id"] for op in openings] == list(range(100))
    for op in openings:
        game = CaroGame()
        for m in op["moves"]:
            game.step(m)
        assert not game.is_terminal()


# ---------------------------------------------------------------- lịch thi đấu

def test_schedule_swaps_sides_and_collapses_seeds_for_deterministic_pairs(tmp_path):
    cfg = _cfg(tmp_path, {
        "heuristic": {"type": "heuristic"},
        "alphabeta": {"type": "alphabeta", "params": {"depth": 1, "fast_eval": True}},
    })
    tasks = build_schedule(cfg, load_openings(cfg["openings_file"]))
    assert len(tasks) == 2 * 2  # 2 khai cuộc x 2 lượt, chỉ seed đầu vì cả hai tất định
    for op_id in (0, 1):
        sides = {(t.agent_x.label, t.agent_o.label) for t in tasks if t.opening_id == op_id}
        assert sides == {("heuristic", "alphabeta"), ("alphabeta", "heuristic")}
    assert len({t.game_id for t in tasks}) == len(tasks)


def test_schedule_runs_all_seeds_for_stochastic_pairs(tmp_path):
    cfg = _cfg(tmp_path, {"random": {"type": "random"}, "heuristic": {"type": "heuristic"},
                          "mcts": {"type": "mcts"}})
    tasks = build_schedule(cfg, load_openings(cfg["openings_file"]))
    # 3 cặp đều có tác tử ngẫu nhiên: 3 cặp x 3 seed x 2 khai cuộc x 2 lượt
    assert len(tasks) == 3 * 3 * 2 * 2
    assert len({(t.seed_x, t.seed_o, t.game_id) for t in tasks}) == len(tasks)


def test_load_config_validates_modes(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nmode: B\nbudget: 0.5\nagents: {a: {type: random}, b: {type: random}}\n",
                    encoding="utf-8")
    assert load_config(path)["budget"] is None  # chế độ B luôn dùng cấu hình mặc định
    path.write_text("name: x\nmode: A\nagents: {a: {type: random}, b: {type: random}}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_repo_arena_configs_are_valid():
    for name in ("smoke", "mode_a", "mode_b"):
        cfg = load_config(ROOT / "configs" / "arena" / f"{name}.yaml")
        assert cfg["openings_file"] == "configs/openings.json"
    assert load_config(ROOT / "configs" / "arena" / "mode_a.yaml")["num_openings"] == 100


# ---------------------------------------------------------------- chơi ván

def _task(x: AgentSpec, o: AgentSpec, opening=(60, 61), budget=0.01) -> GameTask:
    return GameTask(game_id="t", experiment="test", mode="A", budget=budget, size=11, rule="vietnam",
                    opening_id=0, opening_moves=tuple(opening), seed=0,
                    agent_x=x, agent_o=o, seed_x=1, seed_o=2)


def test_play_game_record_has_required_fields_and_is_reproducible():
    x = AgentSpec("random", "random", stochastic=True)
    o = AgentSpec("heuristic", "heuristic")
    r1 = play_game(_task(x, o))
    r2 = play_game(_task(x, o))
    assert REQUIRED_FIELDS <= set(r1)
    assert [m["action"] for m in r1["moves"]] == [m["action"] for m in r2["moves"]]  # cùng seed, cùng ván
    assert r1["result"] in (-1, 0, 1)
    assert r1["opening_moves"] == [60, 61]
    assert r1["num_moves"] == 2 + len(r1["moves"])
    assert all({"player", "action", "time_ms", "stats"} <= set(m) for m in r1["moves"])
    # khai cuộc 2 nước -> X đi tiếp, rồi luân phiên
    assert [m["player"] for m in r1["moves"][:3]] == [1, -1, 1]
    assert r1["agent_x"]["seed"] == 1 and r1["agent_o"]["seed"] == 2
    assert r1["winner"] in ("random", "heuristic", None)


def test_play_game_logs_search_stats():
    x = AgentSpec("ab", "alphabeta", {"depth": 1, "fast_eval": True})
    o = AgentSpec("mcts", "mcts", {"simulations": 30}, stochastic=True)
    rec = play_game(_task(x, o, budget=None))
    ab_stats = [m["stats"] for m in rec["moves"] if m["player"] == 1]
    mcts_stats = [m["stats"] for m in rec["moves"] if m["player"] == -1]
    assert all("nodes" in s and "depth_reached" in s for s in ab_stats)
    assert all("sims" in s for s in mcts_stats)


class _IllegalAgent(Agent):
    name = "illegal"

    def select_action(self, state, budget=None):
        return 60  # luôn đánh vào ô đã có quân của khai cuộc


def test_illegal_move_forfeits_and_is_logged(monkeypatch):
    monkeypatch.setitem(agent_factory.AGENT_TYPES, "illegal", _IllegalAgent)
    rec = play_game(_task(AgentSpec("bad", "illegal"), AgentSpec("random", "random")))
    assert rec["result"] == -1 and rec["winner"] == "random"
    assert rec["error"] and "60" in rec["error"]


def test_build_agent_passes_seed_only_when_supported():
    assert build_agent(AgentSpec("r", "random"), seed=5).seed == 5
    build_agent(AgentSpec("ab", "alphabeta", {"depth": 1}), seed=5)  # không nhận seed, không lỗi


# ---------------------------------------------------------------- chạy giải

def test_run_tournament_writes_log_and_resumes(tmp_path):
    cfg = _cfg(tmp_path, {"random": {"type": "random"}, "heuristic": {"type": "heuristic"}},
               seeds=(0,), num_openings=2)
    log = run_tournament(cfg, log_dir=tmp_path / "logs", verbose=False)
    first = load_records(log)
    assert len(first) == 4
    run_tournament(cfg, log_dir=tmp_path / "logs", verbose=False)  # chạy lại: không lặp ván
    assert len(load_records(log)) == 4
    meta = load_records(log.with_suffix(".meta.jsonl"))
    assert len(meta) == 2 and meta[1]["games_this_run"] == 0
    assert meta[0]["hardware"]["cpu_count"]


def test_run_tournament_multiprocess_matches_schedule(tmp_path):
    cfg = _cfg(tmp_path, {"random": {"type": "random"}, "heuristic": {"type": "heuristic"}},
               seeds=(0,), num_openings=2)
    log = run_tournament(cfg, log_dir=tmp_path / "logs", workers=2, verbose=False)
    ids = {r["game_id"] for r in load_records(log)}
    assert ids == {t.game_id for t in build_schedule(cfg, load_openings(cfg["openings_file"]))}
