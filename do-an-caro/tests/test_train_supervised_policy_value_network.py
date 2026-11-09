"""Kiểm thử pipeline huấn luyện Supervised Learning (training/train_supervised_policy_value_network.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.network import PolicyValueNet, load_network
from training.train_supervised_policy_value_network import (
    compute_metrics,
    evaluate,
    load_dataset_split,
    train_one_epoch,
    train_sl,
    train_supervised_policy_value_network,
)


def _create_synthetic_dataset(path: Path, num_samples: int = 16) -> Path:
    """Tạo bộ dữ liệu giả lập có cả 3 split (train, val, test) để test nhanh."""
    np.random.seed(42)
    planes = np.random.randint(0, 2, size=(num_samples, 4, 11, 11), dtype=np.uint8)
    actions = np.random.randint(0, 121, size=(num_samples,), dtype=np.int64)
    values = np.random.uniform(-1.0, 1.0, size=(num_samples,)).astype(np.float32)
    game_ids = np.arange(num_samples, dtype=np.int32)

    np.savez_compressed(
        path,
        train_planes=planes,
        train_actions=actions,
        train_values=values,
        train_game_ids=game_ids,
        val_planes=planes[:4],
        val_actions=actions[:4],
        val_values=values[:4],
        val_game_ids=game_ids[:4],
        test_planes=planes[:4],
        test_actions=actions[:4],
        test_values=values[:4],
        test_game_ids=game_ids[:4],
    )
    return path


def test_load_dataset_split(tmp_path):
    data_path = _create_synthetic_dataset(tmp_path / "dataset.npz", num_samples=16)

    train_ds = load_dataset_split(data_path, "train", augment=False)
    val_ds = load_dataset_split(data_path, "val", augment=False)
    test_ds = load_dataset_split(data_path, "test", augment=False)

    assert len(train_ds) == 16
    assert len(val_ds) == 4
    assert len(test_ds) == 4

    # Kiểm tra sample item
    p, a, v = train_ds[0]
    assert p.shape == (4, 11, 11)
    assert 0 <= a < 121
    assert -1.0 <= v <= 1.0


def test_compute_metrics():
    batch_size = 4
    logits = torch.zeros(batch_size, 121)
    # Gán max logit tại index 5
    logits[:, 5] = 10.0
    target_actions = torch.tensor([5, 5, 10, 20])
    values = torch.zeros(batch_size)
    target_values = torch.zeros(batch_size)

    m = compute_metrics(logits, values, target_actions, target_values)
    assert m["total_samples"] == 4
    assert m["top1_correct"] == 2  # 2 mẫu đoán đúng ô 5
    assert m["value_loss"] == 0.0


def test_train_sl_end_to_end(tmp_path):
    assert train_supervised_policy_value_network is train_sl
    data_path = _create_synthetic_dataset(tmp_path / "data.npz", num_samples=12)
    out_dir = tmp_path / "checkpoints_sl"

    config = {
        "name": "test_sl",
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "blocks": 1,
        "channels": 8,
        "board_size": 11,
        "epochs": 3,
        "batch_size": 4,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "value_weight": 1.0,
        "augment": False,
        "patience": 2,
        "min_delta": 1e-4,
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
    }

    summary = train_sl(config=config, out_dir=out_dir)

    # 1. Kiểm tra các file tạo ra
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "config_used.yaml").exists()

    # 2. Đọc checkpoint tốt nhất bằng load_network
    model, meta = load_network(out_dir / "best.pt", device="cpu")
    assert isinstance(model, PolicyValueNet)
    assert meta["config"] == {"blocks": 1, "channels": 8, "size": 11}
    assert meta["seed"] == 42
    assert "epoch" in meta
    assert "val_metrics" in meta
    assert "engine_version" in meta

    # 3. Kiểm tra log
    lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3  # Chạy đủ 3 epochs
    rec = json.loads(lines[0])
    assert "train" in rec and "val" in rec
    assert rec["epoch"] == 1


def test_train_sl_early_stopping(tmp_path):
    data_path = _create_synthetic_dataset(tmp_path / "data_es.npz", num_samples=8)
    out_dir = tmp_path / "checkpoints_es"

    # Đặt min_delta cao và patience=1 để kích hoạt early stopping
    config = {
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "blocks": 1,
        "channels": 8,
        "board_size": 11,
        "epochs": 10,
        "batch_size": 4,
        "lr": 0.0000001,  # Học rất chậm để val_loss không cải thiện
        "patience": 1,
        "min_delta": 10.0,  # Ngưỡng cải thiện bất khả thi -> dừng ngay sau 1 epoch
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
    }

    summary = train_sl(config=config, out_dir=out_dir)
    assert summary["epochs_ran"] <= 3  # Dừng sớm thay vì chạy cả 10 epochs

