"""Huấn luyện mạng Policy-Value bằng Supervised Learning (Cấp 5).

Tuân thủ chặt chẽ Proposal mục 5 và Issue #10:
- Kiến trúc mạng: `PolicyValueNet(blocks=6, channels=64, size=11)` từ `agents.network`.
- Mask ô bận: gán `-1e9` cho các ô đã có quân trước khi tính cross-entropy và top-k.
- Hàm mất mát: Cross-Entropy (policy) + MSE (value) + L2 (weight decay trong Adam).
- Optimizer: Adam, lr 1e-3, batch size 256.
- Early stopping & Auto-resume: lưu `best_val_loss`, `best_epoch`, `patience_counter`
  trong `latest.pt`; tự chạy tiếp nếu `out_dir/latest.pt` đã tồn tại để tránh ghi đè `best.pt`.
"""

from __future__ import annotations

import argparse
import json
import math
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

ILLEGAL_LOGIT = -1e9
DEFAULT_OUT_DIR = "checkpoints/level5"


def set_seed(seed: int) -> None:
    """Cố định seed cho tính tái lập hoàn toàn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: dict[str, Any] | None, device: torch.device) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        t_state = rng_state["torch"]
        if isinstance(t_state, torch.Tensor):
            torch.set_rng_state(t_state.cpu())
    if device.type == "cuda" and torch.cuda.is_available() and "cuda" in rng_state:
        cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in rng_state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def mask_illegal_logits(logits: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Gán -1e9 cho các ô đã có quân (mặt phẳng 0 hoặc 1 > 0) trước khi tính loss/top-k."""
    occupied = (planes[:, 0] + planes[:, 1]).flatten(1) > 0
    return logits.masked_fill(occupied, ILLEGAL_LOGIT)


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
        cand = path.parent / f"{path.stem}_{split}.npz"
        if cand.exists():
            path = cand
        else:
            raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {data_path} hoặc {cand}")

    with np.load(path) as loaded:
        keys = loaded.files
        if f"{split}_planes" in keys:
            planes = loaded[f"{split}_planes"]
            actions = loaded[f"{split}_actions"]
            values = loaded[f"{split}_values"]
            game_ids = loaded.get(f"{split}_game_ids")
        elif "planes" in keys:
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
    """Tính các độ đo trên logits (đã mask ô bận): policy loss, value loss, top-1/top-3 acc, MAE."""
    policy_loss = F.cross_entropy(logits, target_actions)
    value_loss = F.mse_loss(values, target_values)
    total_loss = policy_loss + value_weight * value_loss

    preds = logits.argmax(dim=-1)
    top1_correct = (preds == target_actions).sum().item()

    k = min(3, logits.size(-1))
    top3_preds = logits.topk(k, dim=-1).indices
    top3_correct = (top3_preds == target_actions.unsqueeze(-1)).any(dim=-1).sum().item()

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
    """Đánh giá mô hình trên tập validation hoặc test (có mask ô bận)."""
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
            logits = mask_illegal_logits(logits, planes)
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
    """Huấn luyện 1 epoch (mask ô bận trước khi tính cross-entropy và top-k)."""
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
        logits = mask_illegal_logits(logits, planes)

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


def _sync_log_file(log_path: Path, max_epoch: int) -> list[dict[str, Any]]:
    """Giữ các dòng log hợp lệ đến `max_epoch`, loại bỏ trùng lặp và ghi lại nguyên tử."""
    if not log_path.exists():
        return []

    by_epoch: dict[int, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = int(rec.get("epoch", 0))
        if 1 <= ep <= max_epoch:
            by_epoch[ep] = rec

    history = [by_epoch[ep] for ep in sorted(by_epoch)]
    tmp = log_path.with_suffix(".tmp")
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history)
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, log_path)
    return history


def _resolve_best_state(
    target_out_dir: Path,
    payload: dict[str, Any],
    device: torch.device,
) -> tuple[float, int]:
    """Khôi phục `best_val_loss` và `best_epoch` thật sự để không bao giờ ghi đè `best.pt` bằng mạng kém hơn."""
    best_val_loss = payload.get("best_val_loss")
    best_epoch = int(payload.get("best_epoch", 0))

    if best_val_loss is None:
        # Tương thích checkpoint cũ chưa lưu trường best_val_loss
        val_m = payload.get("val_metrics") or {}
        best_val_loss = float(val_m["loss"]) if "loss" in val_m else float("inf")
        if "loss" in val_m and best_epoch == 0:
            best_epoch = int(payload.get("epoch") or 0)
    else:
        best_val_loss = float(best_val_loss)

    best_path = target_out_dir / "best.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        best_ckpt_loss = (best_payload.get("val_metrics") or {}).get("loss")
        if best_ckpt_loss is not None and float(best_ckpt_loss) < best_val_loss:
            best_val_loss = float(best_ckpt_loss)
            best_epoch = int(best_payload.get("epoch") or best_epoch)

    return best_val_loss, best_epoch


def train_sl(
    config: dict[str, Any],
    out_dir: str | Path | None = None,
    resume: str | Path | None = None,
    no_resume: bool = False,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Huấn luyện Supervised Learning đầy đủ, tự động chạy tiếp nếu đã có `out_dir/latest.pt`."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    target_out_dir = Path(out_dir or config.get("out_dir", DEFAULT_OUT_DIR))
    target_out_dir.mkdir(parents=True, exist_ok=True)

    dev_name = device_name or config.get("device", "auto")
    device = pick_device(dev_name)
    print("=" * 60)
    print("Bắt đầu huấn luyện Supervised Learning (Cấp 5)")
    print(f"Thiết bị: {device} | Engine: v{ENGINE_VERSION} | Seed: {seed}")
    print(f"Thư mục lưu: {target_out_dir}")
    print("=" * 60)

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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Khởi tạo PolicyValueNet: blocks={blocks}, channels={channels}, size={board_size} ({total_params:,} tham số)")

    # 3. Optimizer & Siêu tham số
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    value_weight = float(config.get("value_weight", 1.0))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 1e-4))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Tự động tiếp tục từ latest.pt nếu có (hoặc từ --resume)
    log_path = target_out_dir / "train_log.jsonl"
    latest_path = target_out_dir / "latest.pt"

    if no_resume:
        resume_path = None
    elif resume is not None:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint resume: {resume_path}")
    elif latest_path.exists():
        resume_path = latest_path
    else:
        resume_path = None

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    training_history: list[dict[str, Any]] = []

    if resume_path is not None:
        print(f"Phát hiện checkpoint {resume_path}, tiếp tục huấn luyện...")
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        last_epoch = int(payload.get("epoch") or 0)
        start_epoch = last_epoch + 1
        best_val_loss, best_epoch = _resolve_best_state(target_out_dir, payload, device)
        patience_counter = int(payload.get("patience_counter", 0))
        _restore_rng_state(payload.get("rng_state"), device)
        training_history = _sync_log_file(log_path, last_epoch)
        print(
            f"  Tiếp tục từ epoch {start_epoch} | best_val_loss={best_val_loss:.4f} "
            f"(epoch {best_epoch}) | patience={patience_counter}/{patience}"
        )
    else:
        # Chạy mới hoàn toàn: xóa log cũ nếu có để tránh trùng epoch
        if log_path.exists():
            log_path.unlink()

    config_saved = dict(config)
    config_saved.update({
        "out_dir": str(target_out_dir),
        "device": str(device),
        "engine_version": ENGINE_VERSION,
        "total_params": total_params,
    })
    (target_out_dir / "config_used.yaml").write_text(yaml.dump(config_saved), encoding="utf-8")

    start_train_time = time.perf_counter()

    if patience_counter >= patience:
        print(f"\n[Resume] Huấn luyện đã dừng sớm trước đó (patience {patience_counter}/{patience}). Giữ nguyên best.pt.")
    elif start_epoch > epochs:
        print(f"\n[Resume] Đã hoàn tất đủ {epochs} epochs trước đó. Giữ nguyên best.pt.")
    else:
        print(f"\nBắt đầu từ epoch {start_epoch} đến {epochs} (patience={patience}, min_delta={min_delta}):")
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'P-Loss':>8} | {'V-Loss':>8} | {'Top-1':>7} | {'Val Loss':>10} | {'Val P-L':>8} | {'Val V-L':>8} | {'Val Top1':>8} | {'Time':>6}")
        print("-" * 95)

        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()

            train_metrics = train_one_epoch(model, train_loader, optimizer, device, value_weight)
            val_metrics = evaluate(model, val_loader, device, value_weight)

            epoch_time = time.perf_counter() - epoch_start
            val_loss = val_metrics["loss"]

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

            improved = val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
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

            # Lưu latest.pt SAU khi cập nhật best_val_loss và patience_counter
            save_checkpoint(
                target_out_dir / "latest.pt",
                model,
                epoch=epoch,
                val_metrics=val_metrics,
                seed=seed,
                optimizer_state=optimizer.state_dict(),
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                rng_state=_capture_rng_state(device),
            )

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "time_seconds": round(epoch_time, 2),
            }
            training_history.append(record)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        "best_val_loss": round(best_val_loss, 4) if math.isfinite(best_val_loss) else None,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "total_training_time_seconds": round(total_training_time, 2),
        "test_metrics": test_metrics,
        "epochs_ran": len(training_history),
        "best_checkpoint": str(best_path),
    }
    tmp_summary = target_out_dir / "summary.tmp"
    tmp_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_summary, target_out_dir / "summary.json")
    return summary


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
    parser.add_argument("--resume", help="Đường dẫn checkpoint chỉ định để tiếp tục huấn luyện.")
    parser.add_argument("--no-resume", action="store_true", help="Bỏ qua latest.pt và huấn luyện lại từ đầu.")
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

    for k, v in vars(args).items():
        if v is not None and k not in ("config", "resume", "no_resume"):
            config[k] = v

    if "data_path" not in config:
        print("Lỗi: Cần cung cấp đường dẫn dữ liệu qua file config hoặc cờ `--data`.", file=sys.stderr)
        sys.exit(1)

    train_sl(
        config=config,
        out_dir=args.out_dir,
        resume=args.resume,
        no_resume=args.no_resume,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
