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
    actor_critic_update,
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


def test_train_rl_no_resume_cleans_pool_and_rounds(tmp_path):
    """Chạy --no-resume với mạng SL mới trên cùng out_dir phải dọn sạch pool/rounds và thay anchor_sl.pt."""
    def _make_marked_sl(name: str, bias_val: float) -> Path:
        net = PolicyValueNet(blocks=1, channels=8, size=11)
        with torch.no_grad():
            net.policy_fc.bias.fill_(bias_val)
        return save_checkpoint(tmp_path / f"{name}.pt", net, epoch=1)

    sl_a = _make_marked_sl("sl_a", 0.111)
    sl_b = _make_marked_sl("sl_b", 0.999)
    out_dir = tmp_path / "level6_shared"

    common = {
        "board_size": 11,
        "rule": "vietnam",
        "episodes_per_round": 4,
        "rollout_batch_games": 4,
        "minibatch_size": 8,
        "eval_games": 2,
        "opening_min_moves": 2,
        "opening_max_moves": 2,
        "device": "cpu",
        "seed": 1,
    }

    # Lần 1: chạy 2 vòng (8 episodes) với mạng SL A
    train_rl(config={**common, "episodes": 8}, out_dir=out_dir, init_checkpoint=sl_a)
    assert (out_dir / "rounds" / "round_002.pt").exists()
    assert (out_dir / "pool" / "round_002.pt").exists()
    anchor_a, _ = load_network(out_dir / "pool" / "anchor_sl.pt")
    assert torch.allclose(anchor_a.policy_fc.bias, torch.full_like(anchor_a.policy_fc.bias, 0.111))

    # Lần 2: chạy lại --no-resume với mạng SL B trên cùng out_dir (chỉ 1 vòng)
    train_rl(config={**common, "episodes": 4}, out_dir=out_dir, init_checkpoint=sl_b, no_resume=True)

    # Các file round_002 của lần 1 phải bị xóa hoàn toàn
    assert not (out_dir / "rounds" / "round_002.pt").exists()
    assert not (out_dir / "pool" / "round_002.pt").exists()
    assert [p.name for p in sorted((out_dir / "rounds").glob("round_*.pt"))] == ["round_000.pt", "round_001.pt"]

    # Bản neo trong pool phải là mạng SL B (0.999), không còn SL A (0.111)
    anchor_b, _ = load_network(out_dir / "pool" / "anchor_sl.pt")
    assert torch.allclose(anchor_b.policy_fc.bias, torch.full_like(anchor_b.policy_fc.bias, 0.999))

    # Lần 3: auto-resume tiếp lên 8 episodes phải giữ bản neo SL B và log [0, 1, 2]
    train_rl(config={**common, "episodes": 8}, out_dir=out_dir, init_checkpoint=sl_b)
    anchor_after, _ = load_network(out_dir / "pool" / "anchor_sl.pt")
    assert torch.allclose(anchor_after.policy_fc.bias, torch.full_like(anchor_after.policy_fc.bias, 0.999))
    log_lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l)["round"] for l in log_lines] == [0, 1, 2]


def test_train_rl_best_not_overwritten_on_tie(tmp_path, monkeypatch):
    """Chỉ cập nhật best.pt khi score_vs_sl lớn hơn hẳn best_score, không ghi đè khi hòa điểm."""
    import training.train_rl as rl_mod

    sl_ckpt = _make_level5_checkpoint(tmp_path)
    out_dir = tmp_path / "level6_tie"

    scores = iter([0.50, 0.75, 0.75])  # vòng 0: 0.50, vòng 1: 0.75 (best!), vòng 2: 0.75 (hòa -> không ghi đè)
    monkeypatch.setattr(
        rl_mod,
        "evaluate_vs_baseline",
        lambda *args, **kwargs: (lambda s: {"games": 100, "wins": int(s * 100), "draws": 0, "losses": 100 - int(s * 100), "score": s, "ci95": [s - 0.09, s + 0.09]})(next(scores)),
    )

    train_rl(
        config={
            "episodes": 8,
            "episodes_per_round": 4,
            "rollout_batch_games": 4,
            "minibatch_size": 8,
            "eval_games": 100,
            "opening_min_moves": 2,
            "opening_max_moves": 2,
            "device": "cpu",
            "seed": 0,
        },
        out_dir=out_dir,
        init_checkpoint=sl_ckpt,
    )

    _, best_meta = load_network(out_dir / "best.pt")
    assert best_meta["epoch"] == 1  # giữ nguyên vòng 1, không bị vòng 2 hòa 0.75 ghi đè
    log_recs = [json.loads(l) for l in (out_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["is_best"] for r in log_recs] == [True, True, False]


def test_actor_critic_update_kl_and_value_warmup():
    torch.manual_seed(0)
    sl_net = PolicyValueNet(blocks=1, channels=8, size=11)
    model = PolicyValueNet(blocks=1, channels=8, size=11)
    model.load_state_dict(sl_net.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(11)

    planes = rng.integers(0, 2, size=(16, 4, 11, 11)).astype(np.float32)
    planes[:, :2, :2, :] = 0.0  # giữ 22 ô đầu trống để actions 0..15 luôn hợp lệ
    actions = np.arange(16, dtype=np.int64)
    returns = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(16)], dtype=np.float32)

    policy_w_before = model.policy_fc.weight.detach().clone()
    stem_bn_before = model.stem[1].running_mean.detach().clone()
    value_w_before = model.value_fc[0].weight.detach().clone()

    # Warmup chỉ train đầu value: backbone + policy phải giữ nguyên tuyệt đối, policy_loss == 0
    warmup_metrics = actor_critic_update(
        model, opt, planes, actions, returns,
        {"minibatch_size": 8, "kl_coef": 0.05, "normalize_advantage": True, "augment": False},
        torch.device("cpu"), rng,
        sl_model=sl_net, value_only=True,
    )
    assert warmup_metrics["policy_loss"] == 0.0
    assert warmup_metrics["kl"] <= 1e-6
    assert torch.equal(model.policy_fc.weight, policy_w_before)
    assert torch.equal(model.stem[1].running_mean, stem_bn_before)
    assert not torch.equal(model.value_fc[0].weight, value_w_before)

    # Bật cập nhật đầy đủ: trọng số policy thay đổi và KL >= 0
    full_metrics = actor_critic_update(
        model, opt, planes, actions, returns,
        {"minibatch_size": 8, "kl_coef": 0.05, "normalize_advantage": True, "augment": False},
        torch.device("cpu"), rng,
        sl_model=sl_net, value_only=False,
    )
    assert full_metrics["kl"] >= 0.0
    assert not torch.equal(model.policy_fc.weight, policy_w_before)


def test_rl_config_stabilized_and_logs_kl_and_warmup(tmp_path):
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "training" / "rl.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["lr"] == pytest.approx(1e-5)
    assert cfg["temperature"] == pytest.approx(1.0)
    assert cfg["opponent_temperature"] == pytest.approx(1.0)
    assert cfg["kl_coef"] == pytest.approx(0.05)
    assert cfg["normalize_advantage"] is True

    sl_ckpt = _make_level5_checkpoint(tmp_path)
    out_dir = tmp_path / "level6_kl"
    train_rl(
        config={
            "episodes": 8,
            "episodes_per_round": 4,
            "rollout_batch_games": 4,
            "minibatch_size": 8,
            "lr": 1e-5,
            "kl_coef": 0.05,
            "opponent_temperature": 1.0,
            "normalize_advantage": True,
            "value_warmup_rounds": 1,
            "eval_games": 2,
            "opening_min_moves": 2,
            "opening_max_moves": 2,
            "device": "cpu",
            "seed": 3,
        },
        out_dir=out_dir,
        init_checkpoint=sl_ckpt,
    )
    recs = [json.loads(l) for l in (out_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(recs) == 3
    assert recs[0]["kl"] is None
    assert recs[1]["policy_loss"] == 0.0
    assert isinstance(recs[1]["kl"], float) and recs[1]["kl"] >= 0.0
    assert isinstance(recs[2]["kl"], float) and recs[2]["kl"] >= 0.0

    # Vòng 1 chỉ warmup đầu value nên đầu policy vẫn khớp bản neo SL
    sl_net, _ = load_network(sl_ckpt)
    r1_net, _ = load_network(out_dir / "rounds" / "round_001.pt")
    assert torch.equal(r1_net.policy_fc.weight, sl_net.policy_fc.weight)
