"""Dataset management, symmetries, and deduplication for Caro supervised learning.

Supports:
- 8 symmetry transformations of 4-plane board representation and action index
- Canonical form reduction and deduplication via byte hashing
- Train/Validation/Test 70/10/20 splitting by game (preventing data leakage)
- Compressed dataset serialization (.npz)
- Dataset wrapper compatible with numpy and PyTorch DataLoader
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Try importing PyTorch Dataset, fallback to object if PyTorch not installed yet
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    TorchDataset = object  # type: ignore
    HAS_TORCH = False


def transform_coords(r: int, c: int, size: int, k: int, flip: bool) -> tuple[int, int]:
    """Chuyển đổi toạ độ (r, c) qua phép xoay k*90 độ ngược chiều kim đồng hồ và lật ngang."""
    if k == 0:
        nr, nc = r, c
    elif k == 1:
        nr, nc = size - 1 - c, r
    elif k == 2:
        nr, nc = size - 1 - r, size - 1 - c
    elif k == 3:
        nr, nc = c, size - 1 - r
    else:
        raise ValueError(f"k phải nằm trong khoảng 0..3, nhận {k}")

    if flip:
        nc = size - 1 - nc

    return nr, nc


def apply_symmetry(planes: np.ndarray, action: int, k: int, flip: bool) -> tuple[np.ndarray, int]:
    """Áp dụng phép đối xứng (xoay k*90 độ, lật nếu flip=True) lên 4 mặt phẳng và nước đi.

    Args:
        planes: Mảng shape (4, size, size)
        action: Chỉ số ô 0..size*size-1
        k: Số lần xoay 90 độ (0, 1, 2, 3)
        flip: Có lật ngang hay không

    Returns:
        (transformed_planes, transformed_action)
    """
    size = planes.shape[1]
    out = np.zeros_like(planes)

    # Mặt phẳng 0 (quân mình), 1 (quân đối thủ), 2 (nước vừa đi) được xoay/lật
    for c in range(3):
        rot = np.rot90(planes[c], k)
        out[c] = np.fliplr(rot) if flip else rot

    # Mặt phẳng 3 là lượt đi (toàn 1.0 hoặc toàn 0.0), bất biến qua phép đối xứng
    out[3] = planes[3]

    r, c = divmod(action, size)
    nr, nc = transform_coords(r, c, size, k, flip)
    new_action = nr * size + nc

    return out, new_action


def get_all_symmetries(
    planes: np.ndarray,
    action: int,
    value: float
) -> list[tuple[np.ndarray, int, float]]:
    """Trả về 8 biến thể đối xứng của bộ (planes, action, value)."""
    symmetries: list[tuple[np.ndarray, int, float]] = []
    for k in range(4):
        for flip in (False, True):
            tp, ta = apply_symmetry(planes, action, k, flip)
            symmetries.append((tp, ta, value))
    return symmetries


def canonical_form(
    planes: np.ndarray,
    action: int,
    value: float
) -> tuple[bytes, np.ndarray, int, float]:
    """Chuẩn hoá thế cờ về dạng chuẩn tắc nhỏ nhất qua 8 phép đối xứng.

    Trả về:
        (canonical_key, canonical_planes, canonical_action, value)
    """
    variants: list[tuple[bytes, int, np.ndarray, float]] = []
    for k in range(4):
        for flip in (False, True):
            tp, ta = apply_symmetry(planes, action, k, flip)
            # Đại diện thế cờ từ góc nhìn người đến lượt
            combined = (tp[0] - tp[1]).astype(np.int8)
            key = (
                combined.tobytes()
                + tp[2].astype(np.int8).tobytes()
                + tp[3].astype(np.int8).tobytes()
            )
            variants.append((key, ta, tp, value))

    # Sắp xếp theo (key, action) để đảm bảo tất định 100% kể cả khi thế cờ tự đối xứng
    variants.sort(key=lambda x: (x[0], x[1]))
    best_key, best_action, best_planes, best_value = variants[0]
    return best_key, best_planes, best_action, best_value


@dataclass(frozen=True)
class TrainingSample:
    """Một mẫu huấn luyện (thế cờ, nhãn policy, nhãn value)."""

    planes: np.ndarray          # shape (4, size, size), dtype uint8 hoặc float32
    action: int                 # 0..size*size-1
    value: float                # +1.0 (thắng), -1.0 (thua), 0.0 (hoà)
    game_id: int                # ID ván cờ nguồn
    step: int                   # Số thứ tự nước đi trong ván


@dataclass
class GameRecord:
    """Dữ liệu một ván cờ tự chơi."""

    game_id: int
    samples: list[TrainingSample] = field(default_factory=list)
    winner: int | None = None
    move_count: int = 0
    opening_moves: list[int] = field(default_factory=list)
    duration: float = 0.0


def deduplicate_samples(
    samples: list[TrainingSample],
    canonicalize: bool = True,
    average_values: bool = True,
    exclude_keys: set[bytes] | None = None,
) -> tuple[list[TrainingSample], dict[str, Any]]:
    """Loại bỏ các thế cờ trùng nhau qua 8 phép đối xứng.

    Args:
        samples: Danh sách các mẫu ban đầu
        canonicalize: Nếu True, chuyển mẫu về dạng chuẩn tắc canonical
        average_values: Nếu True, lấy trung bình value của các bản trùng lặp
        exclude_keys: Tập các khoá chuẩn tắc cần loại bỏ (tránh rò rỉ từ tập khác)

    Returns:
        (unique_samples, stats) với stats chứa raw_count, unique_count, duplicates_count,
        excluded_count, và seen_keys.
    """
    seen_keys: set[bytes] = set()
    key_to_sample: dict[bytes, TrainingSample] = {}
    key_to_values: dict[bytes, list[float]] = {}
    duplicates_count = 0
    excluded_count = 0

    for s in samples:
        key, c_planes, c_action, c_val = canonical_form(s.planes, s.action, s.value)
        if exclude_keys is not None and key in exclude_keys:
            excluded_count += 1
            continue

        if key in seen_keys:
            duplicates_count += 1
            if average_values:
                key_to_values[key].append(c_val)
        else:
            seen_keys.add(key)
            stored = TrainingSample(
                planes=c_planes if canonicalize else s.planes,
                action=c_action if canonicalize else s.action,
                value=c_val if canonicalize else s.value,
                game_id=s.game_id,
                step=s.step,
            )
            key_to_sample[key] = stored
            if average_values:
                key_to_values[key] = [c_val]

    if average_values:
        unique_samples = []
        for key, s in key_to_sample.items():
            vals = key_to_values[key]
            avg_val = round(float(np.mean(vals)), 4) if len(vals) > 1 else s.value
            unique_samples.append(TrainingSample(
                planes=s.planes,
                action=s.action,
                value=avg_val,
                game_id=s.game_id,
                step=s.step,
            ))
    else:
        unique_samples = list(key_to_sample.values())

    stats = {
        "raw_count": len(samples),
        "unique_count": len(unique_samples),
        "duplicates_count": duplicates_count,
        "excluded_count": excluded_count,
        "seen_keys": seen_keys,
    }
    return unique_samples, stats


def split_games(
    games: list[GameRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[GameRecord], list[GameRecord], list[GameRecord]]:
    """Chia danh sách ván theo tỷ lệ train/val/test (mặc định 70/10/20).

    Chia THEO VÁN (không theo thế cờ) để chống rò rỉ dữ liệu giữa các tập.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Tổng tỷ lệ chia phải bằng 1.0, nhận {total_ratio}")

    rng = random.Random(seed)
    shuffled = list(games)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    # Đảm bảo tổng khớp n_total
    if n_train + n_val > n_total:
        n_train = max(0, n_total - n_val)

    train_games = shuffled[:n_train]
    val_games = shuffled[n_train:n_train + n_val]
    test_games = shuffled[n_train + n_val:]

    return train_games, val_games, test_games


def samples_to_arrays(samples: list[TrainingSample]) -> dict[str, np.ndarray]:
    """Chuyển đổi danh sách TrainingSample thành các mảng NumPy."""
    if not samples:
        return {
            "planes": np.empty((0, 4, 11, 11), dtype=np.uint8),
            "actions": np.empty((0,), dtype=np.int64),
            "values": np.empty((0,), dtype=np.float32),
            "game_ids": np.empty((0,), dtype=np.int32),
        }

    planes = np.stack([s.planes.astype(np.uint8) for s in samples])
    actions = np.array([s.action for s in samples], dtype=np.int64)
    values = np.array([s.value for s in samples], dtype=np.float32)
    game_ids = np.array([s.game_id for s in samples], dtype=np.int32)

    return {
        "planes": planes,
        "actions": actions,
        "values": values,
        "game_ids": game_ids,
    }


def save_dataset(dataset_dict: dict[str, Any], path: str | Path) -> Path:
    """Lưu dữ liệu dạng file nén .npz."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset_dict)
    return out_path


def load_dataset(path: str | Path, split: str | None = None) -> dict[str, np.ndarray]:
    """Đọc dữ liệu từ file .npz.

    Nếu split được chỉ định ("train", "val", "test"), trả về dict với các key
    "planes", "actions", "values", "game_ids" tương ứng của split đó.
    """
    with np.load(path) as loaded:
        if split is not None:
            prefix = f"{split}_"
            return {
                "planes": loaded[f"{prefix}planes"].copy(),
                "actions": loaded[f"{prefix}actions"].copy(),
                "values": loaded[f"{prefix}values"].copy(),
                "game_ids": loaded[f"{prefix}game_ids"].copy(),
            }
        return {k: loaded[k].copy() for k in loaded.files}


def save_part_file(part_path: str | Path, games: list[GameRecord]) -> Path:
    """Lưu một phần (chunk) các ván cờ vào file .npz để phục vụ checkpoint và chạy tiếp."""
    out_path = Path(part_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_samples = [s for g in games for s in g.samples]
    arrays = samples_to_arrays(all_samples)
    opening_flat = [m for g in games for m in g.opening_moves]
    game_meta = {
        "meta_game_ids": np.array([g.game_id for g in games], dtype=np.int32),
        "meta_winners": np.array([0 if g.winner is None else g.winner for g in games], dtype=np.int8),
        "meta_move_counts": np.array([g.move_count for g in games], dtype=np.int32),
        "meta_durations": np.array([g.duration for g in games], dtype=np.float32),
        "meta_opening_flat": np.array(opening_flat, dtype=np.int16),
        "meta_opening_lens": np.array([len(g.opening_moves) for g in games], dtype=np.int16),
    }
    np.savez_compressed(out_path, **arrays, **game_meta)
    return out_path


def load_part_file(part_path: str | Path) -> list[GameRecord]:
    """Nạp một file chunk .npz và tái tạo danh sách GameRecord."""
    with np.load(part_path) as loaded:
        planes = loaded["planes"].copy()
        actions = loaded["actions"].copy()
        values = loaded["values"].copy()
        sample_gids = loaded["game_ids"].copy()
        meta_gids = loaded["meta_game_ids"].copy()
        meta_winners = loaded["meta_winners"].copy()
        meta_moves = loaded["meta_move_counts"].copy()
        meta_dur = loaded["meta_durations"].copy()
        if "meta_opening_flat" in loaded and "meta_opening_lens" in loaded:
            opening_flat = loaded["meta_opening_flat"].copy()
            opening_lens = loaded["meta_opening_lens"].copy()
            op_offset = 0
            openings: list[list[int]] = []
            for l in opening_lens:
                openings.append(opening_flat[op_offset:op_offset + l].tolist())
                op_offset += l
        else:
            openings = [[] for _ in meta_gids]

    records: list[GameRecord] = []
    sample_idx = 0
    total_samples = len(sample_gids)

    for idx, gid in enumerate(meta_gids):
        g_samples: list[TrainingSample] = []
        while sample_idx < total_samples and sample_gids[sample_idx] == gid:
            g_samples.append(TrainingSample(
                planes=planes[sample_idx],
                action=int(actions[sample_idx]),
                value=float(values[sample_idx]),
                game_id=int(gid),
                step=len(g_samples) + 1,
            ))
            sample_idx += 1
        records.append(GameRecord(
            game_id=int(gid),
            samples=g_samples,
            winner=int(meta_winners[idx]),
            move_count=int(meta_moves[idx]),
            duration=float(meta_dur[idx]),
            opening_moves=openings[idx],
        ))
    return records


class CaroDataset(TorchDataset):
    """Lớp Dataset chứa dữ liệu Caro 11x11, tương thích cả NumPy và PyTorch DataLoader."""

    def __init__(
        self,
        planes: np.ndarray,
        actions: np.ndarray,
        values: np.ndarray,
        game_ids: np.ndarray | None = None,
        augment: bool = False,
        to_tensor: bool = False,
    ) -> None:
        super().__init__()
        self.planes = planes.astype(np.uint8)
        self.actions = actions.astype(np.int64)
        self.values = values.astype(np.float32)
        self.game_ids = game_ids
        self.augment = augment
        self.to_tensor = to_tensor

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        planes = self.planes[idx]
        action = int(self.actions[idx])
        value = float(self.values[idx])

        # Tăng cường dữ liệu bằng 8 phép đối xứng ngẫu nhiên khi train
        if self.augment:
            k = random.randint(0, 3)
            flip = random.choice([False, True])
            planes, action = apply_symmetry(planes, action, k, flip)

        planes_out = planes.astype(np.float32)

        if self.to_tensor and HAS_TORCH:
            return (
                torch.from_numpy(planes_out).float(),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(value, dtype=torch.float32),
            )

        return planes_out, action, value

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        split: str = "train",
        augment: bool = False,
        to_tensor: bool = False,
    ) -> CaroDataset:
        """Khởi tạo CaroDataset trực tiếp từ file .npz."""
        data = load_dataset(path, split=split)
        return cls(
            planes=data["planes"],
            actions=data["actions"],
            values=data["values"],
            game_ids=data.get("game_ids"),
            augment=augment,
            to_tensor=to_tensor,
        )

