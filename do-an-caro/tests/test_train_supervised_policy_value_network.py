"""Kiểm thử pipeline huấn luyện Supervised Learning (training/train_supervised_policy_value_network.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.network import PolicyValueNet, load_network, save_checkpoint
from training.train_supervised_policy_value_network import (
    ILLEGAL_LOGIT,
    compute_metrics,
    evaluate,
    load_dataset_split,
    mask_illegal_logits,
    train_sl,
)


def _create_synthetic_dataset(path: Path, num_samples: int = 16, seed: int = 42) -> Path:
    """Tạo bộ dữ liệu giả lập hợp lệ (action luôn nằm trên ô trống) có đủ 3 split."""
    rng = np.random.default_rng(seed)

    def _make_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        planes = np.zeros((n, 4, 11, 11), dtype=np.uint8)
        actions = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            # Chọn ngẫu nhiên 10-30 ô bận (không trùng nhau giữa mặt phẳng 0 và 1)
            occupied_idx = rng.choice(121, size=20, replace=False)
            half = 10
            for idx in occupied_idx[:half]:
                r, c = divmod(int(idx), 11)
                planes[i, 0, r, c] = 1
            for idx in occupied_idx[half:]:
                r, c = divmod(int(idx), 11)
                planes[i, 1, r, c] = 1
            # Đặt nước vừa đi và lượt đi
            lr, lc = divmod(int(occupied_idx[0]), 11)
            planes[i, 2, lr, lc] = 1
            planes[i, 3].fill(i % 2)
            # Chọn action từ các ô còn trống
            empty_cells = np.setdiff1d(np.arange(121), occupied_idx)
            actions[i] = int(rng.choice(empty_cells))
        values = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
        game_ids = np.arange(n, dtype=np.int32)
        return planes, actions, values, game_ids

    tp, ta, tv, tg = _make_split(num_samples)
    vp, va, vv, vg = _make_split(max(4, num_samples // 2))
    xp, xa, xv, xg = _make_split(max(4, num_samples // 2))

    np.savez_compressed(
        path,
        train_planes=tp,
        train_actions=ta,
        train_values=tv,
        train_game_ids=tg,
        val_planes=vp,
        val_actions=va,
        val_values=vv,
        val_game_ids=vg,
        test_planes=xp,
        test_actions=xa,
        test_values=xv,
        test_game_ids=xg,
    )
    return path


def test_load_dataset_split(tmp_path):
    data_path = _create_synthetic_dataset(tmp_path / "dataset.npz", num_samples=16)

    train_ds = load_dataset_split(data_path, "train", augment=False)
    val_ds = load_dataset_split(data_path, "val", augment=False)
    test_ds = load_dataset_split(data_path, "test", augment=False)

    assert len(train_ds) == 16
    assert len(val_ds) == 8
    assert len(test_ds) == 8

    p, a, v = train_ds[0]
    assert p.shape == (4, 11, 11)
    assert 0 <= a < 121
    assert -1.0 <= v <= 1.0


def test_mask_illegal_logits_blocks_occupied_cells():
    planes = torch.zeros(2, 4, 11, 11, dtype=torch.float32)
    # Mẫu 0: ô 0 có quân mình, ô 5 có quân đối thủ; ô 10 trống
    planes[0, 0, 0, 0] = 1.0
    planes[0, 1, 0, 5] = 1.0
    # Mẫu 1: ô 120 có quân đối thủ
    planes[1, 1, 10, 10] = 1.0

    raw_logits = torch.zeros(2, 121, dtype=torch.float32)
    raw_logits[0, 0] = 100.0   # ô bận có logit rất lớn
    raw_logits[0, 5] = 50.0    # ô bận có logit lớn
    raw_logits[0, 10] = 2.0    # ô trống tốt nhất
    raw_logits[1, 120] = 99.0  # ô bận
    raw_logits[1, 1] = 3.0     # ô trống cao nhất mẫu 1 (khác target 2)

    masked = mask_illegal_logits(raw_logits, planes)
    assert masked[0, 0].item() == pytest.approx(ILLEGAL_LOGIT)
    assert masked[0, 5].item() == pytest.approx(ILLEGAL_LOGIT)
    assert masked[0, 10].item() == pytest.approx(2.0)
    assert masked[1, 120].item() == pytest.approx(ILLEGAL_LOGIT)

    # Top-1 phải chọn ô trống 10 ở mẫu 0, và ô 1 ở mẫu 1 (không chọn ô bận 0, 5 hay 120)
    m = compute_metrics(
        masked,
        torch.zeros(2),
        torch.tensor([10, 2]),
        torch.zeros(2),
    )
    assert m["top1_correct"] == 1  # mẫu 0 đoán đúng ô 10, mẫu 1 đoán ô 1 != 2
    assert torch.isfinite(torch.tensor(m["policy_loss"]))

    # Kiểm tra evaluate() thực sự mask ô bận của model output
    net = PolicyValueNet(blocks=1, channels=4, size=11)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.policy_fc.bias[0] = 100.0  # ô 0 bị chiếm ở mẫu 0
        net.policy_fc.bias[10] = 5.0   # ô 10 trống và là nhãn đúng
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(
        TensorDataset(planes[:1], torch.tensor([10]), torch.tensor([0.0])),
        batch_size=1,
    )
    eval_m = evaluate(net, loader, torch.device("cpu"))
    assert eval_m["top1_acc"] == pytest.approx(1.0)


def test_compute_metrics():
    batch_size = 4
    logits = torch.zeros(batch_size, 121)
    logits[:, 5] = 10.0
    target_actions = torch.tensor([5, 5, 10, 20])
    values = torch.zeros(batch_size)
    target_values = torch.zeros(batch_size)

    m = compute_metrics(logits, values, target_actions, target_values)
    assert m["total_samples"] == 4
    assert m["top1_correct"] == 2
    assert m["value_loss"] == 0.0


def test_train_sl_end_to_end(tmp_path):
    data_path = _create_synthetic_dataset(tmp_path / "data.npz", num_samples=12)
    out_dir = tmp_path / "level5"

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
        "patience": 5,
        "min_delta": 1e-4,
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
    }

    summary = train_sl(config=config, out_dir=out_dir)

    assert (out_dir / "best.pt").exists()
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "config_used.yaml").exists()

    model, meta = load_network(out_dir / "best.pt", device="cpu")
    assert isinstance(model, PolicyValueNet)
    assert meta["config"] == {"blocks": 1, "channels": 8, "size": 11}
    assert meta["seed"] == 42
    assert "epoch" in meta and "val_metrics" in meta and "engine_version" in meta

    # latest.pt phải lưu best_val_loss và patience_counter
    _, latest_meta = load_network(out_dir / "latest.pt", device="cpu")
    assert "best_val_loss" in latest_meta
    assert "patience_counter" in latest_meta
    assert latest_meta["epoch"] == 3

    lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert summary["epochs_ran"] == 3


def test_train_sl_auto_resume_preserves_best_and_log(tmp_path):
    data_path = _create_synthetic_dataset(tmp_path / "data_resume.npz", num_samples=12)
    out_dir = tmp_path / "level5_resume"

    base_cfg = {
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "blocks": 1,
        "channels": 8,
        "board_size": 11,
        "epochs": 2,
        "batch_size": 4,
        "lr": 0.005,
        "patience": 3,
        "min_delta": 1e-4,
        "augment": False,
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
    }

    # Chạy 2 epochs đầu
    train_sl(config=base_cfg, out_dir=out_dir)

    # Giả lập best.pt trước đó có loss cực tốt (0.001) ở epoch 1 và latest.pt đang có patience_counter=1
    # để kiểm tra khi tự động chạy tiếp, epoch 3 không được ghi đè best.pt và patience tăng tiếp lên 2 -> 3
    best_model, _ = load_network(out_dir / "best.pt", device="cpu")
    sentinel_bias = 0.12345
    with torch.no_grad():
        best_model.policy_fc.bias.fill_(sentinel_bias)
    save_checkpoint(
        out_dir / "best.pt",
        best_model,
        epoch=1,
        val_metrics={"loss": 0.001},
        seed=42,
    )

    latest_model, latest_meta = load_network(out_dir / "latest.pt", device="cpu")
    save_checkpoint(
        out_dir / "latest.pt",
        latest_model,
        epoch=2,
        val_metrics=latest_meta["val_metrics"],
        seed=42,
        optimizer_state=latest_meta.get("optimizer_state"),
        best_val_loss=0.001,
        best_epoch=1,
        patience_counter=1,
    )

    # Giả lập Colab bị ngắt và chạy lại đúng lệnh cũ (không truyền --resume, tăng epochs=6)
    resumed_cfg = {**base_cfg, "epochs": 6}
    summary2 = train_sl(config=resumed_cfg, out_dir=out_dir)

    # Phải giữ nguyên best.pt cũ (không bị ghi đè bởi mạng epoch 3/4 kém hơn)
    loaded_best, best_meta = load_network(out_dir / "best.pt", device="cpu")
    assert best_meta["epoch"] == 1
    assert best_meta["val_metrics"]["loss"] == pytest.approx(0.001)
    assert torch.allclose(loaded_best.policy_fc.bias, torch.full_like(loaded_best.policy_fc.bias, sentinel_bias))

    # patience=3, đã có patience_counter=1 trước đó -> chỉ chạy thêm 2 epochs (epoch 3, 4) là early stop
    assert summary2["epochs_ran"] == 4
    lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().split("\n")
    logged_epochs = [json.loads(l)["epoch"] for l in lines]
    assert logged_epochs == [1, 2, 3, 4]  # Không trùng số epoch

    # Chạy lại một lần nữa sau khi đã early stop -> phải bỏ qua vòng lặp ngay lập tức
    summary3 = train_sl(config=resumed_cfg, out_dir=out_dir)
    assert summary3["epochs_ran"] == 4
    lines_after = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines_after) == 4


def test_train_sl_early_stopping(tmp_path):
    data_path = _create_synthetic_dataset(tmp_path / "data_es.npz", num_samples=8)
    out_dir = tmp_path / "checkpoints_es"

    config = {
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "blocks": 1,
        "channels": 8,
        "board_size": 11,
        "epochs": 10,
        "batch_size": 4,
        "lr": 0.0000001,
        "patience": 1,
        "min_delta": 10.0,
        "augment": False,
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
    }

    summary = train_sl(config=config, out_dir=out_dir)
    assert summary["epochs_ran"] <= 3
