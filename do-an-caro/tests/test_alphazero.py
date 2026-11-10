"""Kiểm thử vòng lặp AlphaZero-inspired (training/alphazero.py): chạy trọn vòng và chạy tiếp khi bị ngắt."""

from __future__ import annotations

import json
from pathlib import Path

from arena.logger import load_records
from training.alphazero import AlphaZeroRunner, _merge, DEFAULTS, load_config

ROOT = Path(__file__).resolve().parent.parent


def _tiny_cfg(out: Path) -> dict:
    return _merge(DEFAULTS, {
        "out_dir": str(out),
        "device": "cpu",
        "network": {"blocks": 1, "channels": 8},
        "selfplay": {"games_per_round": 2, "chunk_games": 1, "workers": 1, "simulations": 4, "temp_moves": 4},
        "train": {"epochs": 1, "batch_size": 32},
        "gating": {"games": 2, "simulations": 4, "openings_file": str(ROOT / "configs" / "openings.json")},
    })


def test_full_round_writes_checkpoints_buffer_and_log(tmp_path):
    out = tmp_path / "az"
    state = AlphaZeroRunner(_tiny_cfg(out), log=False).run(rounds=1)
    assert state["round"] == 2 and state["phase"] == "selfplay"
    for name in ("best.pt", "current.pt", "state.json", "meta.jsonl", "rounds/round_001.pt"):
        assert (out / name).exists(), name
    assert len(list((out / "buffer").glob("r001_g*.npz"))) == 2  # 2 ván, mỗi file 1 ván
    log = load_records(out / "train_log.jsonl")
    assert len(log) == 1
    row = log[0]
    assert row["round"] == 1 and row["selfplay_games"] == 2 and row["gate_games"] == 2
    assert {"policy_loss", "value_loss", "gate_score", "gate_ci95", "accepted"} <= set(row)


def test_resume_after_interruption(tmp_path):
    out = tmp_path / "az"
    cfg = _tiny_cfg(out)
    AlphaZeroRunner(cfg, log=False).run(rounds=1)
    # "Bị ngắt" ngay đầu vòng 2: hết giờ trước khi chơi ván nào
    state = AlphaZeroRunner(cfg, log=False).run(rounds=2, max_hours=0)
    assert state["round"] == 2 and state["games_done"] == 0
    # Giả lập ngắt giữa vòng: đã lưu 1/2 ván của vòng 2
    st = json.loads((out / "state.json").read_text())
    runner = AlphaZeroRunner(cfg, log=False)
    runner.cfg["selfplay"]["games_per_round"] = 1
    runner.selfplay(st, max_hours=None)
    runner.cfg["selfplay"]["games_per_round"] = 2
    assert st["games_done"] == 1
    runner.save_state(st)
    # Chạy lại: chơi tiếp đúng 1 ván còn thiếu rồi train, gating
    state = AlphaZeroRunner(cfg, log=False).run(rounds=2)
    assert state["round"] == 3
    assert len(list((out / "buffer").glob("r002_g*.npz"))) == 2
    assert [r["round"] for r in load_records(out / "train_log.jsonl")] == [1, 2]
    # Chạy lại lần nữa: không làm thêm gì
    AlphaZeroRunner(cfg, log=False).run(rounds=2)
    assert len(load_records(out / "train_log.jsonl")) == 2


def test_repo_configs_load():
    cfg = load_config(ROOT / "configs" / "alphazero.yaml")
    assert cfg["network"] == {"blocks": 4, "channels": 64}
    assert cfg["gating"]["threshold"] == 0.55 and cfg["buffer_rounds"] == 4
    assert cfg["out_dir"] == "checkpoints/alphazero2" and cfg["rounds"] == 12
    assert cfg["selfplay"]["games_per_round"] == 200 and cfg["train"]["epochs"] == 5
    assert cfg["selfplay"]["radius"] == 2 and cfg["selfplay"]["fpu"] == 0.25
    smoke = load_config(ROOT / "configs" / "alphazero_smoke.yaml")
    assert smoke["selfplay"]["games_per_round"] == 8 and smoke["selfplay"]["c_puct"] == 1.5
    assert smoke["selfplay"]["radius"] == 2  # lấy từ DEFAULTS
