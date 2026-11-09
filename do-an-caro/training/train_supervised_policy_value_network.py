"""Huấn luyện mạng Policy-Value bằng Supervised Learning (Cấp 5).

Tuân thủ chặt chẽ Proposal mục 5 và Issue #10:
- Kiến trúc mạng: `PolicyValueNet(blocks=6, channels=64, size=11)` từ `agents.network`.
- Nhãn:
  - Policy: nước Alpha-Beta chọn (0..120).
  - Value: kết quả cuối ván theo góc nhìn người đến lượt (+1 thắng, -1 thua, 0 hòa).
- Dữ liệu: Phân chia 70/10/20 theo ván, không theo thế cờ để chống rò rỉ.
  Hỗ trợ tăng cường dữ liệu bằng 8 phép đối xứng ngẫu nhiên khi train.
- Hàm mất mát:
  Loss = Cross-Entropy (policy) + MSE (value) + L2 (weight decay trong Adam).
- Optimizer: Adam, lr 1e-3, batch size 256.
- Early stopping: Dừng sớm theo validation loss, lưu `best.pt`.
- Checkpoint: Định dạng chuẩn `{"model_state", "config", "epoch", "val_metrics", "seed", "engine_version"}`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from agents.network import (
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    PolicyValueNet,
    pick_device,
    save_checkpoint,
)
from game.constants import DEFAULT_BOARD_SIZE, ENGINE_VERSION
from training.dataset import CaroDataset


def set_seed(seed: int) -> None:
    """Cố định seed cho tính tái lập hoàn toàn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_split(
    data_path: str | Path,
    split: str,
    augment: bool = False,
) -> CaroDataset:
    """Nạp dataset cho một split ('train', 'val', hoặc 'test').

    Hỗ trợ cả file gộp (`dataset_sl.npz` chứa `train_*`, `val_*`, `test_*`)
    và file riêng lẻ (`dataset_sl_train.npz`).
    """
    path = Path(data_path)
    if not path.exists():
        # Thử tìm theo hậu tố
        cand = path.parent / f"{path.stem}_{split}.npz"
        if cand.exists():
            path = cand
        else:
            raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {data_path} hoặc {cand}")

    # Nếu là file riêng lẻ có tên chứa split hoặc chứa keys không có tiền tố
    with np.load(path) as loaded:
        keys = loaded.files
        if f"{split}_planes" in keys:
            # File gộp
            planes = loaded[f"{split}_planes"]
            actions = loaded[f"{split}_actions"]
            values = loaded[f"{split}_values"]
            game_ids = loaded.get(f"{split}_game_ids")
        elif "planes" in keys:
            # File riêng lẻ
            planes = loaded["planes"]
            actions = loaded["actions"]
            values = loaded["values"]
            game_ids = loaded.get("game_ids")
        else:
            raise KeyError(f"File {path} không chứa dữ liệu hợp lệ cho split '{split}' (có keys: {keys})")

    return CaroDataset(
        planes=planes,
        actions=actions,
        values=values,
        game_ids=game_ids,
        augment=augment,
        to_tensor=True,
    )


def compute_metrics(
    logits: torch.Tensor,
    values: torch.Tensor,
    target_actions: torch.Tensor,
    target_values: torch.Tensor,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """Tính các độ đo: loss policy, loss value, top-1 accuracy, top-3 accuracy, value MSE/MAE."""
    policy_loss = F.cross_entropy(logits, target_actions)
    value_loss = F.mse_loss(values, target_values)
    total_loss = policy_loss + value_weight * value_loss

    # Top-1 accuracy
    preds = logits.argmax(dim=-1)
    top1_correct = (preds == target_actions).sum().item()

    # Top-3 accuracy
    k = min(3, logits.size(-1))
    top3_preds = logits.topk(k, dim=-1).indices
    top3_correct = (top3_preds == target_actions.unsqueeze(-1)).any(dim=-1).sum().item()

    # Value MAE
    value_mae = F.l1_loss(values, target_values).item()

    batch_size = target_actions.size(0)
    return {
        "loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "top1_correct": float(top1_correct),
        "top3_correct": float(top3_correct),
        "value_mae_sum": float(value_mae * batch_size),
        "total_samples": float(batch_size),
    }


def evaluate(
    model: PolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """Đánh giá mô hình trên tập validation hoặc test."""
    model.eval()
    total_samples = 0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_value_mae = 0.0

    with torch.no_grad():
        for planes, actions, values in loader:
            planes = planes.to(device)
            actions = actions.to(device)
            values = values.to(device)

            logits, pred_values = model(planes)
            m = compute_metrics(logits, pred_values, actions, values, value_weight)

            bs = int(m["total_samples"])
            total_samples += bs
            total_loss += m["loss"] * bs
            total_policy_loss += m["policy_loss"] * bs
            total_value_loss += m["value_loss"] * bs
            total_top1 += int(m["top1_correct"])
            total_top3 += int(m["top3_correct"])
            total_value_mae += m["value_mae_sum"]

    if total_samples == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "top1_acc": 0.0,
            "top3_acc": 0.0,
            "value_mae": 0.0,
            "samples": 0,
        }

    return {
        "loss": total_loss / total_samples,
        "policy_loss": total_policy_loss / total_samples,
        "value_loss": total_value_loss / total_samples,
        "top1_acc": total_top1 / total_samples,
        "top3_acc": total_top3 / total_samples,
        "value_mae": total_value_mae / total_samples,
        "samples": total_samples,
    }


def train_one_epoch(
    model: PolicyValueNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """Huấn luyện 1 epoch."""
    model.train()
    total_samples = 0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_value_mae = 0.0

    for planes, actions, values in loader:
        planes = planes.to(device)
        actions = actions.to(device)
        values = values.to(device)

        optimizer.zero_grad()
        logits, pred_values = model(planes)

        policy_loss = F.cross_entropy(logits, actions)
        value_loss = F.mse_loss(pred_values, values)
        loss = policy_loss + value_weight * value_loss

        loss.backward()
        optimizer.step()

        m = compute_metrics(logits.detach(), pred_values.detach(), actions, values, value_weight)
        bs = int(m["total_samples"])
        total_samples += bs
        total_loss += m["loss"] * bs
        total_policy_loss += m["policy_loss"] * bs
        total_value_loss += m["value_loss"] * bs
        total_top1 += int(m["top1_correct"])
        total_top3 += int(m["top3_correct"])
        total_value_mae += m["value_mae_sum"]

    return {
        "loss": total_loss / total_samples if total_samples else 0.0,
        "policy_loss": total_policy_loss / total_samples if total_samples else 0.0,
        "value_loss": total_value_loss / total_samples if total_samples else 0.0,
        "top1_acc": total_top1 / total_samples if total_samples else 0.0,
        "top3_acc": total_top3 / total_samples if total_samples else 0.0,
        "value_mae": total_value_mae / total_samples if total_samples else 0.0,
        "samples": total_samples,
    }


def train_sl(
    config: dict[str, Any],
    out_dir: str | Path | None = None,
    resume: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Hàm chạy huấn luyện Supervised Learning đầy đủ theo cấu hình."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Thư mục xuất checkpoint và logs
    target_out_dir = Path(out_dir or config.get("out_dir", "checkpoints/supervised-policy-value-network"))
    target_out_dir.mkdir(parents=True, exist_ok=True)

    # Thiết bị phần cứng
    dev_name = device_name or config.get("device", "auto")
    device = pick_device(dev_name)
    print(f"============================================================")
    print(f"Bắt đầu huấn luyện Supervised Learning (Cấp 5)")
    print(f"Thiết bị: {device} | Engine: v{ENGINE_VERSION} | Seed: {seed}")
    print(f"Thư mục lưu: {target_out_dir}")
    print(f"============================================================")

    # 1. Nạp dữ liệu
    data_path = config["data_path"]
    batch_size = int(config.get("batch_size", 256))
    augment = bool(config.get("augment", True))
    num_workers = int(config.get("num_workers", 0))

    print(f"Đang nạp tập Train, Val, Test từ: {data_path} (augment={augment})...")
    train_dataset = load_dataset_split(data_path, "train", augment=augment)
    val_dataset = load_dataset_split(data_path, "val", augment=False)
    has_test = True
    try:
        test_dataset = load_dataset_split(data_path, "test", augment=False)
    except Exception:
        has_test = False
        test_dataset = None

    print(f"  - Tập Train: {len(train_dataset):,} mẫu")
    print(f"  - Tập Val:   {len(val_dataset):,} mẫu")
    if has_test and test_dataset is not None:
        print(f"  - Tập Test:  {len(test_dataset):,} mẫu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if has_test and test_dataset is not None
        else None
    )

    # 2. Khởi tạo mô hình mạng
    blocks = int(config.get("blocks", DEFAULT_BLOCKS))
    channels = int(config.get("channels", DEFAULT_CHANNELS))
    board_size = int(config.get("board_size", DEFAULT_BOARD_SIZE))

    model = PolicyValueNet(blocks=blocks, channels=channels, size=board_size).to(device)
    print(f"Khởi tạo PolicyValueNet: blocks={blocks}, channels={channels}, size={board_size}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Số lượng tham số mô hình: {total_params:,}")

    # 3. Optimizer & Siêu tham số
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    value_weight = float(config.get("value_weight", 1.0))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 1e-4))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Hỗ trợ tiếp tục huấn luyện (resume)
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            print(f"Đang tiếp tục huấn luyện từ checkpoint: {resume_path}...")
            payload = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            if "optimizer_state" in payload:
                optimizer.load_state_dict(payload["optimizer_state"])
            start_epoch = payload.get("epoch", 0) + 1
            if "val_metrics" in payload and "loss" in payload["val_metrics"]:
                best_val_loss = payload["val_metrics"]["loss"]
            print(f"  Tiếp tục từ epoch {start_epoch}, best_val_loss trước đó: {best_val_loss:.4f}")

    # Lưu lại config để tái lập hoàn toàn
    config_saved = dict(config)
    config_saved.update({
        "out_dir": str(target_out_dir),
        "device": str(device),
        "engine_version": ENGINE_VERSION,
        "total_params": total_params,
    })
    (target_out_dir / "config_used.yaml").write_text(yaml.dump(config_saved), encoding="utf-8")

    log_path = target_out_dir / "train_log.jsonl"
    training_history: list[dict[str, Any]] = []

    print(f"\nBắt đầu chạy {epochs} epochs (Early stopping patience={patience}, min_delta={min_delta}):")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'P-Loss':>8} | {'V-Loss':>8} | {'Top-1':>7} | {'Val Loss':>10} | {'Val P-L':>8} | {'Val V-L':>8} | {'Val Top1':>8} | {'Time':>6}")
    print("-" * 95)

    start_train_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, value_weight)
        val_metrics = evaluate(model, val_loader, device, value_weight)

        epoch_time = time.perf_counter() - epoch_start
        val_loss = val_metrics["loss"]

        # In tiến độ gọn gàng
        print(
            f"{epoch:5d} | "
            f"{train_metrics['loss']:10.4f} | "
            f"{train_metrics['policy_loss']:8.4f} | "
            f"{train_metrics['value_loss']:8.4f} | "
            f"{train_metrics['top1_acc']*100:6.1f}% | "
            f"{val_metrics['loss']:10.4f} | "
            f"{val_metrics['policy_loss']:8.4f} | "
            f"{val_metrics['value_loss']:8.4f} | "
            f"{val_metrics['top1_acc']*100:7.1f}% | "
            f"{epoch_time:5.1f}s"
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time_seconds": round(epoch_time, 2),
        }
        training_history.append(record)

        # Ghi log jsonl
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Lưu latest checkpoint mỗi epoch
        save_checkpoint(
            target_out_dir / "latest.pt",
            model,
            epoch=epoch,
            val_metrics=val_metrics,
            seed=seed,
            optimizer_state=optimizer.state_dict(),
        )

        # Kiểm tra cải thiện và Early Stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                target_out_dir / "best.pt",
                model,
                epoch=epoch,
                val_metrics=val_metrics,
                seed=seed,
            )
            print(f"       -> Lưu checkpoint tốt nhất mới (val_loss={best_val_loss:.4f}) tại epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Đạt ngưỡng kiên nhẫn ({patience} epochs không cải thiện). Dừng sớm tại epoch {epoch}.")
                break

    total_training_time = time.perf_counter() - start_train_time
    print("=" * 95)
    print(f"Hoàn thành huấn luyện trong {total_training_time:.2f} giây.")

    # 5. Đánh giá cuối cùng trên tập Test bằng checkpoint tốt nhất
    best_path = target_out_dir / "best.pt"
    test_metrics = None
    if has_test and test_loader is not None and best_path.exists():
        print(f"\nĐánh giá checkpoint tốt nhất ({best_path}) trên tập Test...")
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        test_metrics = evaluate(model, test_loader, device, value_weight)
        print(f"  - Test Loss:        {test_metrics['loss']:.4f}")
        print(f"  - Test Policy Loss: {test_metrics['policy_loss']:.4f}")
        print(f"  - Test Value Loss:  {test_metrics['value_loss']:.4f}")
        print(f"  - Test Top-1 Acc:   {test_metrics['top1_acc']*100:.2f}%")
        print(f"  - Test Top-3 Acc:   {test_metrics['top3_acc']*100:.2f}%")
        print(f"  - Test Value MAE:   {test_metrics['value_mae']:.4f}")

    summary = {
        "best_val_loss": round(best_val_loss, 4),
        "total_training_time_seconds": round(total_training_time, 2),
        "test_metrics": test_metrics,
        "epochs_ran": len(training_history),
        "best_checkpoint": str(best_path),
    }
    (target_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# Alias hàm huấn luyện theo tên đầy đủ
train_supervised_policy_value_network = train_sl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện Supervised Learning cho Caro 11x11 (Cấp 5).")
    parser.add_argument("config", nargs="?", default=None, help="Đường dẫn file YAML cấu hình.")
    parser.add_argument("--data", dest="data_path", help="Đường dẫn file dữ liệu .npz")
    parser.add_argument("--out-dir", dest="out_dir", help="Thư mục lưu checkpoint và log.")
    parser.add_argument("--epochs", type=int, help="Số epochs huấn luyện.")
    parser.add_argument("--batch-size", type=int, help="Kích thước batch.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--blocks", type=int, help="Số khối ResNet (mặc định 6).")
    parser.add_argument("--channels", type=int, help="Số kênh ResNet (mặc định 64).")
    parser.add_argument("--patience", type=int, help="Số epoch cho early stopping.")
    parser.add_argument("--device", help="Thiết bị ('cuda', 'cpu', hoặc 'auto').")
    parser.add_argument("--seed", type=int, help="Seed ngẫu nhiên.")
    parser.add_argument("--resume", help="Đường dẫn checkpoint để tiếp tục huấn luyện.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = {}

    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        else:
            print(f"Lỗi: Không tìm thấy file config {args.config}", file=sys.stderr)
            sys.exit(1)

    # Ghi đè bởi CLI flags nếu có
    for k, v in vars(args).items():
        if v is not None and k != "config":
            config[k] = v

    if "data_path" not in config:
        print("Lỗi: Cần cung cấp đường dẫn dữ liệu qua file config hoặc cờ `--data`.", file=sys.stderr)
        sys.exit(1)

    train_sl(
        config=config,
        out_dir=args.out_dir,
        resume=args.resume,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

