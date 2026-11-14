"""Thống kê bộ dữ liệu học có giám sát cấp 5 (đề bài mục 3.4) trực tiếp từ data/dataset_sl.npz.

Bổ sung cho dataset_sl_log.json (không còn lưu): số mẫu và số ván theo tập, kiểm tra trùng trong tập
và rò rỉ giữa các tập theo khóa chuẩn tắc (đã chuẩn hóa 8 phép đối xứng khi sinh dữ liệu), phân bố
nhãn value, số quân trên bàn, và hình phân bố nhãn nước đi.

python -m analysis.dataset_stats data/dataset_sl.npz --out results/training/level5/dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
FIG = Path(__file__).resolve().parent / "figures"


def _keys(planes: np.ndarray) -> set[bytes]:
    comb = planes[:, 0].astype(np.int8) - planes[:, 1].astype(np.int8)
    return {comb[i].tobytes() + planes[i, 2].tobytes() + planes[i, 3].tobytes() for i in range(len(planes))}


def dataset_stats(path: Path) -> dict:
    d = np.load(path)
    stats: dict = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "splits": {}}
    keys = {}
    for s in SPLITS:
        p, g, v = d[f"{s}_planes"], d[f"{s}_game_ids"], d[f"{s}_values"]
        keys[s] = _keys(p)
        stones = (p[:, 0].astype(int) + p[:, 1]).reshape(len(p), -1).sum(1)
        games = np.unique(g)
        stats["splits"][s] = {
            "samples": int(len(p)),
            "unique_positions": len(keys[s]),
            "games_with_samples": int(len(games)),
            "samples_per_game": round(len(p) / len(games), 2),
            "stones_mean": round(float(stones.mean()), 2),
            "stones_median": float(np.median(stones)),
            "stones_max": int(stones.max()),
            "value_win": int((v > 0.999).sum()),
            "value_loss": int((v < -0.999).sum()),
            "value_draw": int((np.abs(v) < 1e-6).sum()),
            "value_averaged": int(((np.abs(v) > 1e-6) & (np.abs(v) < 0.999)).sum()),
        }
    gids = {s: set(np.unique(d[f"{s}_game_ids"]).tolist()) for s in SPLITS}
    stats["overlap_positions"] = {
        "val_train": len(keys["val"] & keys["train"]),
        "test_train": len(keys["test"] & keys["train"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    stats["overlap_games"] = {
        "val_train": len(gids["val"] & gids["train"]),
        "test_train": len(gids["test"] & gids["train"]),
        "val_test": len(gids["val"] & gids["test"]),
    }
    stats["total_samples"] = sum(stats["splits"][s]["samples"] for s in SPLITS)
    stats["total_games_with_samples"] = len(gids["train"] | gids["val"] | gids["test"])
    return stats


def plot(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for s in SPLITS:
        p = d[f"{s}_planes"]
        stones = (p[:, 0].astype(int) + p[:, 1]).reshape(len(p), -1).sum(1)
        ax.hist(stones, bins=np.arange(0, 62, 2), density=True, histtype="step", lw=1.6, label=f"{s} ({len(p):,})")
    ax.set_xlabel("số quân trên bàn tại thế cờ")
    ax.set_ylabel("tỷ lệ mẫu")
    ax.set_title("Phân bố giai đoạn ván trong dữ liệu cấp 5")
    ax.legend()
    ax = axes[1]
    counts = np.bincount(d["train_actions"], minlength=121).reshape(11, 11)
    im = ax.imshow(counts / counts.sum() * 100, cmap="viridis")
    ax.set_xticks(range(11), [chr(ord("A") + i) for i in range(11)], fontsize=7)
    ax.set_yticks(range(11), [str(i + 1) for i in range(11)], fontsize=7)
    ax.set_title("Nhãn nước đi (train, dạng chuẩn tắc)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% mẫu")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "dataset_level5.png", dpi=200)
    plt.close(fig)
    print("  ✓ dataset_level5.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thống kê bộ dữ liệu cấp 5.")
    parser.add_argument("npz", nargs="?", default="data/dataset_sl.npz")
    parser.add_argument("--out", default="results/training/level5/dataset_stats.json")
    args = parser.parse_args()
    path = Path(args.npz)
    stats = dataset_stats(path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    plot(path)


if __name__ == "__main__":
    main()
