"""Kiểm thử mạng policy-value dùng chung (agents/network.py)."""

from __future__ import annotations

import numpy as np
import torch

from agents.network import (
    NetEvaluator,
    PolicyValueNet,
    encode_game,
    load_checkpoint,
    masked_softmax,
    save_checkpoint,
)
from game.game import CaroGame


def _small_net() -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet(blocks=1, channels=8)


def test_forward_shapes_and_value_range():
    net = _small_net().eval()
    logits, value = net(torch.zeros(3, 4, 11, 11))
    assert logits.shape == (3, 121)
    assert value.shape == (3,)
    assert torch.all(value.abs() <= 1)


def test_masked_softmax_zero_on_occupied_cells():
    logits = np.random.default_rng(0).normal(size=(2, 121))
    mask = np.ones((2, 121), dtype=bool)
    mask[:, :10] = False
    p = masked_softmax(logits, mask)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p[:, :10] == 0)


def test_evaluator_predict_masks_occupied_cells():
    game = CaroGame()
    game.step(60)
    game.step(61)
    priors, value = NetEvaluator(_small_net()).predict(game)
    assert priors[60] == 0 and priors[61] == 0
    assert abs(priors.sum() - 1) < 1e-6
    assert -1 <= value <= 1
    planes, mask = encode_game(game)
    assert planes.shape == (4, 11, 11) and mask.sum() == 119


def test_checkpoint_roundtrip_gives_same_outputs(tmp_path):
    net = _small_net().eval()
    path = save_checkpoint(tmp_path / "net.pt", net, round=3)
    assert not list(tmp_path.glob("*.tmp"))  # ghi file tạm rồi đổi tên
    loaded, meta = load_checkpoint(path)
    assert meta["config"] == {"blocks": 1, "channels": 8, "size": 11}
    assert meta["round"] == 3 and "engine_version" in meta
    x = torch.rand(2, 4, 11, 11)
    with torch.no_grad():
        a, b = net(x), loaded(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
