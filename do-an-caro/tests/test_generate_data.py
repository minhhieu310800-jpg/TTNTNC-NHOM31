"""Unit and integration tests for Caro supervised learning data generation (Issue #9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from game.constants import BLACK, DEFAULT_BOARD_SIZE, WHITE
from game.game import CaroGame
from training.dataset import (
    CaroDataset,
    GameRecord,
    TrainingSample,
    apply_symmetry,
    canonical_form,
    deduplicate_samples,
    get_all_symmetries,
    load_dataset,
    samples_to_arrays,
    save_dataset,
    split_games,
    transform_coords,
)
from training.generate_data import (
    DataGenTask,
    build_tasks,
    generate_dataset,
    play_self_play_game,
)


class TestSymmetries:
    """Kiểm thử 8 phép đối xứng và chuẩn tắc hoá."""

    def test_transform_coords_identity(self):
        r, c = transform_coords(3, 7, 11, k=0, flip=False)
        assert (r, c) == (3, 7)

    def test_transform_coords_four_rotations(self):
        size = 11
        r, c = 2, 8
        for _ in range(4):
            r, c = transform_coords(r, c, size, k=1, flip=False)
        assert (r, c) == (2, 8)

    def test_transform_coords_double_flip(self):
        size = 11
        r, c = 2, 8
        r1, c1 = transform_coords(r, c, size, k=0, flip=True)
        r2, c2 = transform_coords(r1, c1, size, k=0, flip=True)
        assert (r2, c2) == (r, c)

    def test_transform_coords_invalid_k(self):
        with pytest.raises(ValueError):
            transform_coords(0, 0, 11, k=4, flip=False)

    def test_apply_symmetry_planes(self):
        size = 11
        game = CaroGame(size=size)
        game.step(60)  # (5, 5)
        game.step(48)  # (4, 4)
        planes = game.get_planes()
        action = 72    # (6, 6)

        tp, ta = apply_symmetry(planes, action, k=1, flip=False)
        assert tp.shape == (4, size, size)
        # Mặt phẳng 3 (turn) không đổi
        assert np.array_equal(tp[3], planes[3])
        # Nước vừa đi chỉ có 1 điểm duy nhất
        assert tp[2].sum() == 1.0

        ar, ac = divmod(ta, size)
        assert 0 <= ar < size and 0 <= ac < size

    def test_get_all_symmetries_count(self):
        planes = np.zeros((4, 11, 11), dtype=np.float32)
        planes[0, 5, 5] = 1.0
        symmetries = get_all_symmetries(planes, action=60, value=1.0)
        assert len(symmetries) == 8
        for tp, ta, tv in symmetries:
            assert tp.shape == (4, 11, 11)
            assert tv == 1.0

    def test_canonical_form_invariance(self):
        """Các thế cờ xoay/lật của nhau phải cho ra cùng một canonical_key và canonical_action tương ứng."""
        size = 11
        b1 = np.zeros((size, size), dtype=np.int8)
        b1[5, 5] = 1
        b1[4, 4] = -1
        a1 = 6 * size + 6  # (6, 6)

        planes1 = np.zeros((4, size, size), dtype=np.float32)
        planes1[0, 5, 5] = 1.0
        planes1[1, 4, 4] = 1.0
        planes1[2, 4, 4] = 1.0
        planes1[3].fill(1.0)

        # Xoay 90 độ
        planes2, a2 = apply_symmetry(planes1, a1, k=1, flip=False)

        k1, cp1, ca1, cv1 = canonical_form(planes1, a1, 1.0)
        k2, cp2, ca2, cv2 = canonical_form(planes2, a2, 1.0)

        assert k1 == k2
        assert ca1 == ca2
        assert np.array_equal(cp1, cp2)
        assert cv1 == cv2


class TestDeduplication:
    """Kiểm thử loại trùng theo 8 phép đối xứng."""

    def test_deduplicate_identical_and_symmetric_samples(self):
        size = 11
        p = np.zeros((4, size, size), dtype=np.uint8)
        p[0, 5, 5] = 1
        p[1, 4, 4] = 1
        action = 60

        # Tạo mẫu gốc
        s1 = TrainingSample(planes=p, action=action, value=1.0, game_id=0, step=2)

        # Tạo mẫu đối xứng (xoay 90 độ)
        p_rot, a_rot = apply_symmetry(p, action, k=1, flip=False)
        s2 = TrainingSample(planes=p_rot, action=a_rot, value=1.0, game_id=1, step=2)

        # Tạo mẫu đối xứng (lật)
        p_flip, a_flip = apply_symmetry(p, action, k=0, flip=True)
        s3 = TrainingSample(planes=p_flip, action=a_flip, value=1.0, game_id=2, step=2)

        # Mẫu khác biệt hoàn toàn
        p_diff = np.zeros((4, size, size), dtype=np.uint8)
        p_diff[0, 1, 1] = 1
        s4 = TrainingSample(planes=p_diff, action=12, value=-1.0, game_id=3, step=1)

        samples = [s1, s2, s3, s4]
        unique, stats = deduplicate_samples(samples, canonicalize=True)

        assert stats["raw_count"] == 4
        assert stats["unique_count"] == 2
        assert stats["duplicates_count"] == 2
        assert len(unique) == 2


class TestGameSplitting:
    """Kiểm thử chia 70/10/20 theo ván chống rò rỉ dữ liệu."""

    def test_split_games_no_leakage(self):
        games = [
            GameRecord(game_id=i, samples=[], winner=BLACK, move_count=10)
            for i in range(100)
        ]
        train, val, test = split_games(games, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42)

        assert len(train) == 70
        assert len(val) == 10
        assert len(test) == 20

        train_ids = {g.game_id for g in train}
        val_ids = {g.game_id for g in val}
        test_ids = {g.game_id for g in test}

        # Không được có bất kỳ ID ván nào trùng lặp giữa các tập
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert len(train_ids | val_ids | test_ids) == 100

    def test_split_games_invalid_ratios(self):
        games = [GameRecord(game_id=i) for i in range(10)]
        with pytest.raises(ValueError):
            split_games(games, train_ratio=0.5, val_ratio=0.1, test_ratio=0.2)


class TestDataGeneration:
    """Kiểm thử chơi tự chơi và sinh dữ liệu."""

    def test_play_self_play_random_agent(self):
        task = DataGenTask(
            game_id=1,
            seed=123,
            agent_type="random",
            depth=1,
            epsilon=0.1,
            size=11,
            rule="vietnam",
            min_opening_moves=2,
            max_opening_moves=2,
            center_radius=2,
            fast_eval=True,
            radius=2,
            use_pruning=True,
            use_move_ordering=True,
            ordering="cheap",
            use_tt=True,
        )
        record = play_self_play_game(task)
        assert record.game_id == 1
        assert len(record.opening_moves) == 2
        assert len(record.samples) > 0
        assert record.winner in (BLACK, WHITE, 0)

        # Kiểm tra tính hợp lệ của từng mẫu
        for s in record.samples:
            assert s.planes.shape == (4, 11, 11)
            assert 0 <= s.action < 121
            assert s.value in (-1.0, 0.0, 1.0)
            assert s.game_id == 1

    def test_play_self_play_alphabeta_agent(self):
        task = DataGenTask(
            game_id=2,
            seed=456,
            agent_type="alphabeta",
            depth=1,  # độ sâu 1 để test chạy nhanh
            epsilon=0.1,
            size=11,
            rule="vietnam",
            min_opening_moves=2,
            max_opening_moves=2,
            center_radius=2,
            fast_eval=True,
            radius=2,
            use_pruning=True,
            use_move_ordering=True,
            ordering="cheap",
            use_tt=True,
        )
        record = play_self_play_game(task)
        assert record.game_id == 2
        assert len(record.samples) > 0
        assert record.winner in (BLACK, WHITE, 0)

    def test_build_tasks(self):
        cfg = {
            "generation": {
                "num_games": 5,
                "agent_type": "alphabeta",
                "depth": 3,
                "epsilon": 0.1,
                "seed": 999,
            },
            "game": {"size": 11, "rule": "vietnam"},
        }
        tasks = build_tasks(cfg)
        assert len(tasks) == 5
        assert tasks[0].game_id == 0
        assert tasks[4].game_id == 4
        # Mỗi ván phải có seed khác nhau nhưng xác định
        seeds = {t.seed for t in tasks}
        assert len(seeds) == 5


class TestDatasetAndPersistence:
    """Kiểm thử lưu trữ .npz và lớp CaroDataset."""

    def test_save_and_load_dataset(self, tmp_path: Path):
        samples = [
            TrainingSample(
                planes=np.ones((4, 11, 11), dtype=np.uint8),
                action=10,
                value=1.0,
                game_id=0,
                step=1,
            ),
            TrainingSample(
                planes=np.zeros((4, 11, 11), dtype=np.uint8),
                action=20,
                value=-1.0,
                game_id=1,
                step=1,
            ),
        ]
        arrays = samples_to_arrays(samples)
        out_file = tmp_path / "test_data.npz"
        save_dataset({"train_planes": arrays["planes"], "train_actions": arrays["actions"], "train_values": arrays["values"], "train_game_ids": arrays["game_ids"]}, out_file)

        loaded = load_dataset(out_file, split="train")
        assert loaded["planes"].shape == (2, 4, 11, 11)
        assert np.array_equal(loaded["actions"], np.array([10, 20]))
        assert np.array_equal(loaded["values"], np.array([1.0, -1.0]))

        ds = CaroDataset.from_file(out_file, split="train")
        assert len(ds) == 2
        p0, a0, v0 = ds[0]
        assert p0.shape == (4, 11, 11)
        assert a0 == 10
        assert v0 == 1.0

    def test_caro_dataset_augmentation(self):
        planes = np.zeros((4, 11, 11), dtype=np.uint8)
        planes[0, 5, 5] = 1
        actions = np.array([60], dtype=np.int64)
        values = np.array([1.0], dtype=np.float32)

        ds = CaroDataset(planes=np.expand_dims(planes, 0), actions=actions, values=values, augment=True)
        p_aug, a_aug, v_aug = ds[0]
        assert p_aug.shape == (4, 11, 11)
        assert 0 <= a_aug < 121
        assert v_aug == 1.0

    def test_generate_dataset_pipeline_end_to_end(self, tmp_path: Path):
        cfg = {
            "name": "test_pipeline",
            "game": {"size": 11, "rule": "vietnam"},
            "generation": {
                "agent_type": "random",
                "num_games": 6,
                "depth": 1,
                "epsilon": 0.1,
                "min_opening_moves": 2,
                "max_opening_moves": 2,
                "center_radius": 2,
                "seed": 100,
                "workers": 1,
            },
            "split": {"train": 0.7, "val": 0.1, "test": 0.2},
            "deduplication": {"use_symmetries": True},
            "output": {"data_dir": str(tmp_path), "dataset_name": "test_out"},
        }

        log_info = generate_dataset(cfg, out_dir=tmp_path, name="test_out", verbose=False)

        assert log_info["num_games"] == 6
        assert log_info["seed"] == 100
        assert "stats" in log_info
        assert "total_raw_states" in log_info["stats"]
        assert "total_unique_states" in log_info["stats"]

        # Kiểm tra file đã tạo
        npz_path = tmp_path / "test_out.npz"
        log_path = tmp_path / "test_out_log.json"
        assert npz_path.exists()
        assert log_path.exists()

        loaded_json = json.loads(log_path.read_text(encoding="utf-8"))
        assert loaded_json["num_games"] == 6
        assert loaded_json["stats"]["total_unique_states"] <= loaded_json["stats"]["total_raw_states"]
