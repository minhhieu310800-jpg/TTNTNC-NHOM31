"""Kiểm thử pipeline huấn luyện Cấp 6 RL Self-Play Actor-Critic (training/train_rl.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents import RLAgent
from agents.network import PolicyValueNet, load_network, save_checkpoint
from game.game import CaroGame
from tests.test_train_supervised_policy_value_network import _create_synthetic_dataset
from training.dataset import apply_symmetry
from training.train_rl import (
    OpponentPool,
    batch_symmetry,
    sample_batched_actions,
    train_rl,
)
from training.train_supervised_policy_value_network import train_sl


def _make_level5_checkpoint(tmp_path: Path) -> Path:
    data = _create_synthetic_dataset(tmp_path / "sl_data.npz", num_samples=8)
    sl_dir = tmp_path / "level5"
    train_sl(
        config={
            "data_path": str(data),
            "out_dir": str(sl_dir),
            "blocks": 1,
            "channels": 8,
            "board_size": 11,
            "epochs": 1,
            "batch_size": 4,
            "augment": False,
            "device": "cpu",
            "seed": 0,
            "num_workers": 0,
        },
        out_dir=sl_dir,
    )
    return sl_dir / "best.pt"


def test_batch_symmetry_matches_apply_symmetry():
    rng = np.random.default_rng(7)
    planes = rng.integers(0, 2, size=(5, 4, 11, 11), dtype=np.uint8).astype(np.float32)
    # Mặt phẳng 3 biểu diễn lượt đi (toàn 0 hoặc toàn 1) trong Caro
    for i in range(5):
        planes[i, 3].fill(float(i % 2))
    actions = rng.integers(0, 121, size=(5,), dtype=np.int64)

    for k in range(8):
        rot = k % 4
        flip = k >= 4
        bp, ba = batch_symmetry(planes, actions, k, size=11)
        for i in range(len(actions)):
            sp, sa = apply_symmetry(planes[i], int(actions[i]), rot, flip)
            assert np.array_equal(bp[i], sp)
            assert int(ba[i]) == sa


def test_sample_batched_actions_never_picks_occupied():
    torch.manual_seed(0)
    net = PolicyValueNet(blocks=1, channels=8, size=11)
    # Ép logit cực đại vào các ô đã đánh (60, 61, 49) để kiểm tra mask
    with torch.no_grad():
        net.policy_fc.bias.fill_(-5.0)
        for occ in (60, 61, 49):
            net.policy_fc.bias[occ] = 100.0

    game = CaroGame()
    for m in (60, 61, 49):
        game.step(m)

    rng = np.random.default_rng(123)
    games = [game.clone() for _ in range(16)]
    for temp in (0.0, 1.0):
        acts, _ = sample_batched_actions(net, games, temp, torch.device("cpu"), rng)
        for a in acts:
            assert int(a) not in (60, 61, 49)
            assert int(a) in game.legal_actions()


def test_opponent_pool_pins_sl_anchor_and_caps_size(tmp_path):
    net = PolicyValueNet(blocks=1, channels=8, size=11)
    anchor = save_checkpoint(tmp_path / "sl.pt", net, epoch=0)
    pool = OpponentPool(tmp_path / "pool", anchor, max_size=3)

    for rnd in range(1, 6):
        r_ckpt = save_checkpoint(tmp_path / f"round_{rnd:03d}.pt", net, epoch=rnd)
        pool.add_snapshot(r_ckpt, rnd)

    paths = pool.paths()
    assert len(paths) == 3
    assert paths[0].name == "anchor_sl.pt"  # Bản neo SL luôn được giữ
    assert [p.name for p in paths[1:]] == ["round_004.pt", "round_005.pt"]


def test_train_rl_requires_level5_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cấp 5"):
        train_rl(
            config={"episodes": 4, "episodes_per_round": 4, "device": "cpu"},
            out_dir=tmp_path / "level6",
            init_checkpoint=tmp_path / "missing_level5.pt",
        )


def test_train_rl_end_to_end_and_auto_resume(tmp_path):
    sl_ckpt = _make_level5_checkpoint(tmp_path)
    out_dir = tmp_path / "level6"

    cfg = {
        "name": "test_rl",
        "init_checkpoint": str(sl_ckpt),
        "out_dir": str(out_dir),
        "board_size": 11,
        "rule": "vietnam",
        "episodes": 8,
        "episodes_per_round": 4,
        "rollout_batch_games": 4,
        "minibatch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "grad_clip": 1.0,
        "augment": True,
        "temperature": 1.0,
        "opponent_temperature": 0.5,
        "opening_min_moves": 2,
        "opening_max_moves": 2,
        "center_radius": 2,
        "max_pool_size": 4,
        "eval_games": 4,
        "device": "cpu",
        "seed": 42,
    }

    summary1 = train_rl(config=cfg, out_dir=out_dir)
    assert summary1["episodes_done"] == 8
    assert summary1["rounds_done"] == 2
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "pool" / "anchor_sl.pt").exists()
    for r in (0, 1, 2):
        assert (out_dir / "rounds" / f"round_{r:03d}.pt").exists()

    log_lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l)["round"] for l in log_lines] == [0, 1, 2]

    # Nạp checkpoint từ RL bằng load_network và RLAgent
    model, meta = load_network(out_dir / "best.pt", device="cpu")
    assert isinstance(model, PolicyValueNet)
    assert meta["config"] == {"blocks": 1, "channels": 8, "size": 11}
    agent = RLAgent(checkpoint=out_dir / "best.pt", device="cpu")
    g = CaroGame()
    assert agent.select_action(g) in g.legal_actions()

    # Kiểm tra auto-resume khi tăng episodes lên 12 (chỉ chạy thêm vòng 3, không trùng log)
    cfg2 = {**cfg, "episodes": 12}
    summary2 = train_rl(config=cfg2, out_dir=out_dir)
    assert summary2["episodes_done"] == 12
    assert summary2["rounds_done"] == 3
    log_lines2 = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l)["round"] for l in log_lines2] == [0, 1, 2, 3]
