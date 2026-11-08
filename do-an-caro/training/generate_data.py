"""Sinh dữ liệu huấn luyện Supervised Learning từ Alpha-Beta (Issue #9, Proposal mục 5).

Quy trình:
1. Cho Alpha-Beta độ sâu 3 tự chơi với nhau (có nhiễu ε = 0,1 để đa dạng hoá thế cờ).
2. Khởi đầu từ 2-4 nước khai cuộc ngẫu nhiên gần tâm.
3. Gán nhãn: policy = nước Alpha-Beta chọn; value = kết quả cuối ván theo góc nhìn người đến lượt.
4. Chia 70/10/20 THEO VÁN (không theo thế cờ) để chống rò rỉ dữ liệu.
5. Loại trùng theo 8 phép đối xứng của bàn vuông.
6. Ghi log đầy đủ: seed, cấu hình, số ván, số thế cờ trước/sau loại trùng, thời gian sinh.

Sử dụng:
    python -m training.generate_data configs/training/generate_data.yaml [--limit 100] [--workers 4]
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agents.alphabeta_agent import (
    CANDIDATE_RADIUS,
    DEFAULT_DEPTH,
    ORDERING_CHEAP,
    AlphaBetaAgent,
)
from agents.random_agent import RandomAgent
from arena.logger import hardware_info
from game.constants import (
    BLACK,
    DEFAULT_BOARD_SIZE,
    ENGINE_VERSION,
    RULE_VIETNAM,
    WHITE,
)
from game.game import CaroGame
from game.patterns import get_candidate_moves
from training.dataset import (
    GameRecord,
    TrainingSample,
    deduplicate_samples,
    load_part_file,
    samples_to_arrays,
    save_dataset,
    save_part_file,
    split_games,
)

DEFAULT_CONFIG_PATH = "configs/training/generate_data.yaml"
DEFAULT_OUT_DIR = "data"
DEFAULT_DATASET_NAME = "dataset_sl"
DEFAULT_CHUNK_SIZE = 500
PROGRESS_INTERVAL = 25


@dataclass(frozen=True)
class DataGenTask:
    """Nhiệm vụ sinh một ván cờ (gửi được qua tiến trình con)."""

    game_id: int
    seed: int
    agent_type: str
    depth: int
    epsilon: float
    size: int
    rule: str
    min_opening_moves: int
    max_opening_moves: int
    center_radius: int
    fast_eval: bool
    radius: int
    use_pruning: bool
    use_move_ordering: bool
    ordering: str
    use_tt: bool


def play_self_play_game(task: DataGenTask) -> GameRecord:
    """Chơi một ván cờ tự chơi và thu thập dữ liệu huấn luyện.

    Hàm độc lập, không giữ trạng thái toàn cục, chạy an toàn trên Windows spawn worker.
    """
    t0 = time.perf_counter()
    rng = random.Random(task.seed)

    game = CaroGame(size=task.size, rule=task.rule)

    # Khởi tạo tác tử
    if task.agent_type == "alphabeta":
        agent = AlphaBetaAgent(
            depth=task.depth,
            radius=task.radius,
            use_pruning=task.use_pruning,
            use_move_ordering=task.use_move_ordering,
            ordering=task.ordering,
            fast_eval=task.fast_eval,
            use_tt=task.use_tt,
        )
    elif task.agent_type == "random":
        agent = RandomAgent(seed=task.seed)
    else:
        raise ValueError(f"Loại tác tử không hỗ trợ: {task.agent_type}")

    # 1. Khai cuộc ngẫu nhiên gần tâm (2-4 nước)
    opening_moves: list[int] = []
    num_opening = rng.randint(task.min_opening_moves, task.max_opening_moves)
    center = task.size // 2
    r_min = max(0, center - task.center_radius)
    r_max = min(task.size - 1, center + task.center_radius)
    center_cells = [
        r * task.size + c
        for r in range(r_min, r_max + 1)
        for c in range(r_min, r_max + 1)
    ]

    for _ in range(num_opening):
        if game.is_terminal():
            break
        empty_center = [act for act in center_cells if game.board.ravel()[act] == 0]
        if not empty_center:
            empty_center = game.legal_actions()
        if not empty_center:
            break
        op_move = rng.choice(empty_center)
        game.step(op_move)
        opening_moves.append(op_move)

    # 2. Tự chơi với tác tử + epsilon noise
    recorded_steps: list[dict[str, Any]] = []

    while not game.is_terminal():
        curr_player = game.current_player
        planes = game.get_planes().astype(np.uint8)

        # Nhãn policy = nước Alpha-Beta (hoặc tác tử) chọn
        ab_action = int(agent.select_action(game.clone()))

        recorded_steps.append({
            "planes": planes,
            "action": ab_action,
            "player": curr_player,
            "step": game.move_count,
        })

        # Nước đi thực tế trên bàn cờ: với xác suất epsilon chọn ngẫu nhiên để đa dạng
        if rng.random() < task.epsilon:
            candidates = get_candidate_moves(game.board, radius=task.radius)
            if not candidates:
                candidates = game.legal_actions()
            actual_move = rng.choice(candidates)
        else:
            actual_move = ab_action

        game.step(actual_move)

    # 3. Gán nhãn value theo kết quả cuối ván từ góc nhìn người đến lượt
    winner = game.result()
    samples: list[TrainingSample] = []
    for step_data in recorded_steps:
        # +1.0 nếu người đến lượt thắng, -1.0 nếu thua, 0.0 nếu hoà
        val = float(game.get_reward(step_data["player"]))
        samples.append(TrainingSample(
            planes=step_data["planes"],
            action=step_data["action"],
            value=val,
            game_id=task.game_id,
            step=step_data["step"],
        ))

    elapsed = time.perf_counter() - t0
    return GameRecord(
        game_id=task.game_id,
        samples=samples,
        winner=winner,
        move_count=game.move_count,
        opening_moves=opening_moves,
        duration=elapsed,
    )


def _limit_threads() -> None:
    """Giới hạn mỗi worker 1 luồng CPU để tối ưu hiệu năng chạy song song."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def load_generation_config(path: str | Path) -> dict[str, Any]:
    """Đọc cấu hình sinh dữ liệu từ file YAML."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg


def build_tasks(cfg: dict[str, Any], limit: int | None = None) -> list[DataGenTask]:
    """Tạo danh sách nhiệm vụ sinh ván cờ theo cấu hình."""
    gen_cfg = cfg.get("generation", {})
    game_cfg = cfg.get("game", {})

    num_games = int(limit or gen_cfg.get("num_games", 20000))
    agent_type = str(gen_cfg.get("agent_type", "alphabeta"))
    depth = int(gen_cfg.get("depth", DEFAULT_DEPTH))
    epsilon = float(gen_cfg.get("epsilon", 0.1))
    size = int(game_cfg.get("size", DEFAULT_BOARD_SIZE))
    rule = str(game_cfg.get("rule", RULE_VIETNAM))
    min_opening = int(gen_cfg.get("min_opening_moves", 2))
    max_opening = int(gen_cfg.get("max_opening_moves", 4))
    center_radius = int(gen_cfg.get("center_radius", 2))
    fast_eval = bool(gen_cfg.get("fast_eval", True))
    radius = int(gen_cfg.get("radius", CANDIDATE_RADIUS))
    use_pruning = bool(gen_cfg.get("use_pruning", True))
    use_move_ordering = bool(gen_cfg.get("use_move_ordering", True))
    ordering = str(gen_cfg.get("ordering", ORDERING_CHEAP))
    use_tt = bool(gen_cfg.get("use_tt", True))

    base_seed = int(gen_cfg.get("seed", 20261120))

    tasks: list[DataGenTask] = []
    for game_id in range(num_games):
        # Tách seed cho từng ván đảm bảo tính ngẫu nhiên nhưng tái lập hoàn toàn
        task_seed = (base_seed * 10007 + game_id * 37 + 1) % (2**31 - 1)
        tasks.append(DataGenTask(
            game_id=game_id,
            seed=task_seed,
            agent_type=agent_type,
            depth=depth,
            epsilon=epsilon,
            size=size,
            rule=rule,
            min_opening_moves=min_opening,
            max_opening_moves=max_opening,
            center_radius=center_radius,
            fast_eval=fast_eval,
            radius=radius,
            use_pruning=use_pruning,
            use_move_ordering=use_move_ordering,
            ordering=ordering,
            use_tt=use_tt,
        ))
    return tasks


def generate_dataset(
    cfg: dict[str, Any],
    limit: int | None = None,
    workers: int | None = None,
    out_dir: str | Path | None = None,
    name: str | None = None,
    chunk_size: int | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Thực thi pipeline sinh dữ liệu, lưu dần theo chunk, chia 70/10/20 theo ván,

    loại trùng 8 phép đối xứng, chống rò rỉ thế cờ và lưu log.
    """
    start_time = time.perf_counter()

    gen_cfg = cfg.get("generation", {})
    split_cfg = cfg.get("split", {})
    dedup_cfg = cfg.get("deduplication", {})
    out_cfg = cfg.get("output", {})

    output_dir = Path(out_dir or out_cfg.get("data_dir", DEFAULT_OUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = name or out_cfg.get("dataset_name", DEFAULT_DATASET_NAME)

    active_chunk_size = int(
        chunk_size if chunk_size is not None else gen_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)
    )
    parts_dir = Path(out_cfg.get("parts_dir", output_dir / f"{dataset_name}_parts"))
    parts_dir.mkdir(parents=True, exist_ok=True)

    num_workers = int(workers if workers is not None else gen_cfg.get("workers", 1))
    tasks = build_tasks(cfg, limit=limit)
    total_games = len(tasks)

    if verbose:
        print(f"============================================================")
        print(f" Sinh dữ liệu Caro 11x11 cho Supervised Learning (Issue #9)")
        print(f"============================================================")
        print(f"Số ván cần sinh: {total_games}")
        print(f"Tác tử: {tasks[0].agent_type} (độ sâu {tasks[0].depth}, ε = {tasks[0].epsilon})")
        print(f"Số tiến trình (workers): {num_workers}")
        print(f"Lưu checkpoint mỗi: {active_chunk_size} ván -> {parts_dir}")
        print(f"Thư mục lưu: {output_dir}")
        print(f"------------------------------------------------------------")

    # 1. Quét các part file đã có nếu bật resume
    num_chunks = (total_games + active_chunk_size - 1) // active_chunk_size if total_games > 0 else 0
    existing_games_map: dict[int, GameRecord] = {}
    completed_chunks: set[int] = set()

    if resume:
        for c in range(num_chunks):
            part_file = parts_dir / f"part_{c:04d}.npz"
            if part_file.exists():
                try:
                    c_games = load_part_file(part_file)
                    c_start = c * active_chunk_size
                    c_end = min((c + 1) * active_chunk_size, total_games)
                    expected_ids = set(range(c_start, c_end))
                    if {g.game_id for g in c_games} == expected_ids:
                        for g in c_games:
                            existing_games_map[g.game_id] = g
                        completed_chunks.add(c)
                except Exception:
                    pass

    pending_tasks = [t for t in tasks if t.game_id not in existing_games_map]

    if verbose and len(existing_games_map) > 0:
        print(f"  [Resume] Tìm thấy {len(existing_games_map)}/{total_games} ván đã hoàn thành "
              f"từ {len(completed_chunks)} part files. Cần chạy tiếp: {len(pending_tasks)} ván.")

    # Buffer gom các ván của chunk đang chạy dở
    chunk_buffers: dict[int, dict[int, GameRecord]] = {
        c: {} for c in range(num_chunks) if c not in completed_chunks
    }

    finished_count = len(existing_games_map)

    def on_game_finished(rec: GameRecord) -> None:
        nonlocal finished_count
        finished_count += 1
        c = rec.game_id // active_chunk_size
        if c in chunk_buffers:
            chunk_buffers[c][rec.game_id] = rec
            c_start = c * active_chunk_size
            c_end = min((c + 1) * active_chunk_size, total_games)
            expected_count = c_end - c_start
            if len(chunk_buffers[c]) == expected_count:
                part_file = parts_dir / f"part_{c:04d}.npz"
                sorted_chunk = [chunk_buffers[c][gid] for gid in range(c_start, c_end)]
                save_part_file(part_file, sorted_chunk)
                for g in sorted_chunk:
                    existing_games_map[g.game_id] = g
                del chunk_buffers[c]
                completed_chunks.add(c)
                if verbose:
                    print(f"  [Checkpoint] Đã lưu {part_file.name} ({expected_count} ván)")

        if verbose and (finished_count % PROGRESS_INTERVAL == 0 or finished_count == total_games):
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / finished_count if finished_count > 0 else 0
            remaining = (total_games - finished_count) * avg_time
            print(f"  [{finished_count}/{total_games}] ván hoàn thành ({elapsed:.1f}s, dự kiến còn {remaining:.1f}s)")

    if pending_tasks:
        if num_workers <= 1:
            for task in pending_tasks:
                rec = play_self_play_game(task)
                on_game_finished(rec)
        else:
            _limit_threads()
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                futures = [executor.submit(play_self_play_game, t) for t in pending_tasks]
                for fut in as_completed(futures):
                    rec = fut.result()
                    on_game_finished(rec)

    # Lưu nốt các ván còn dư nếu chunk cuối chưa đủ
    for c, buf in list(chunk_buffers.items()):
        if buf:
            part_file = parts_dir / f"part_{c:04d}.npz"
            sorted_chunk = [buf[gid] for gid in sorted(buf.keys())]
            save_part_file(part_file, sorted_chunk)
            for g in sorted_chunk:
                existing_games_map[g.game_id] = g
            if verbose:
                print(f"  [Checkpoint] Đã lưu {part_file.name} ({len(sorted_chunk)} ván)")

    # 3. Tập hợp đầy đủ ván theo đúng thứ tự game_id
    games = [existing_games_map[gid] for gid in range(total_games)]

    # 4. Thống kê kết quả các ván
    black_wins = sum(1 for g in games if g.winner == BLACK)
    white_wins = sum(1 for g in games if g.winner == WHITE)
    draws = sum(1 for g in games if g.winner == 0 or g.winner is None)

    # 5. Chia 70/10/20 theo ván
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.10))
    test_ratio = float(split_cfg.get("test", 0.20))
    base_seed = int(gen_cfg.get("seed", 20261120))

    train_games, val_games, test_games = split_games(
        games,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=base_seed,
    )

    # 6. Thu thập mẫu và loại trùng theo 8 phép đối xứng (chống rò rỉ giữa các tập)
    use_symmetries = bool(dedup_cfg.get("use_symmetries", True))
    average_values = bool(dedup_cfg.get("average_values", True))

    train_raw = [s for g in train_games for s in g.samples]
    val_raw = [s for g in val_games for s in g.samples]
    test_raw = [s for g in test_games for s in g.samples]

    if use_symmetries:
        # Train: loại trùng nội bộ và lấy trung bình value
        train_unique, train_stats = deduplicate_samples(
            train_raw,
            canonicalize=True,
            average_values=average_values,
        )
        train_keys = train_stats["seen_keys"]

        # Val: loại trùng nội bộ, bỏ các thế cờ đã xuất hiện trong train (chống rò rỉ)
        val_unique, val_stats = deduplicate_samples(
            val_raw,
            canonicalize=True,
            average_values=average_values,
            exclude_keys=train_keys,
        )
        val_keys = val_stats["seen_keys"]

        # Test: loại trùng nội bộ, bỏ các thế cờ đã xuất hiện trong train và val (chống rò rỉ)
        test_unique, test_stats = deduplicate_samples(
            test_raw,
            canonicalize=True,
            average_values=average_values,
            exclude_keys=train_keys | val_keys,
        )
    else:
        train_unique = train_raw
        val_unique = val_raw
        test_unique = test_raw
        train_stats = {"raw_count": len(train_raw), "unique_count": len(train_raw), "duplicates_count": 0, "excluded_count": 0}
        val_stats = {"raw_count": len(val_raw), "unique_count": len(val_raw), "duplicates_count": 0, "excluded_count": 0}
        test_stats = {"raw_count": len(test_raw), "unique_count": len(test_raw), "duplicates_count": 0, "excluded_count": 0}

    # Bỏ trường seen_keys (chứa set[bytes]) trước khi serialize sang JSON
    for st in (train_stats, val_stats, test_stats):
        st.pop("seen_keys", None)

    train_stats["leakage_with_train_removed"] = train_stats.get("excluded_count", 0)
    val_stats["leakage_with_train_removed"] = val_stats.get("excluded_count", 0)
    test_stats["leakage_with_train_removed"] = test_stats.get("excluded_count", 0)

    train_arrays = samples_to_arrays(train_unique)
    val_arrays = samples_to_arrays(val_unique)
    test_arrays = samples_to_arrays(test_unique)

    total_raw = train_stats["raw_count"] + val_stats["raw_count"] + test_stats["raw_count"]
    total_unique = train_stats["unique_count"] + val_stats["unique_count"] + test_stats["unique_count"]
    total_duplicates = train_stats["duplicates_count"] + val_stats["duplicates_count"] + test_stats["duplicates_count"]
    total_leakage = val_stats["leakage_with_train_removed"] + test_stats["leakage_with_train_removed"]
    dedup_ratio = (total_duplicates / total_raw) if total_raw > 0 else 0.0

    total_duration = time.perf_counter() - start_time

    # 7. Lưu bộ dữ liệu nén
    dataset_dict: dict[str, Any] = {
        "train_planes": train_arrays["planes"],
        "train_actions": train_arrays["actions"],
        "train_values": train_arrays["values"],
        "train_game_ids": train_arrays["game_ids"],

        "val_planes": val_arrays["planes"],
        "val_actions": val_arrays["actions"],
        "val_values": val_arrays["values"],
        "val_game_ids": val_arrays["game_ids"],

        "test_planes": test_arrays["planes"],
        "test_actions": test_arrays["actions"],
        "test_values": test_arrays["values"],
        "test_game_ids": test_arrays["game_ids"],
    }

    unified_path = output_dir / f"{dataset_name}.npz"
    save_dataset(dataset_dict, unified_path)

    # Lưu thêm các file split riêng cho tiện tải/nạp
    save_dataset(train_arrays, output_dir / f"{dataset_name}_train.npz")
    save_dataset(val_arrays, output_dir / f"{dataset_name}_val.npz")
    save_dataset(test_arrays, output_dir / f"{dataset_name}_test.npz")

    # 8. Ghi log JSON
    log_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": base_seed,
        "engine_version": ENGINE_VERSION,
        "hardware": hardware_info(),
        "config": cfg,
        "num_games": total_games,
        "time_elapsed_seconds": round(total_duration, 2),
        "time_per_game_seconds": round(total_duration / total_games, 4) if total_games > 0 else 0.0,
        "outcomes": {
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "black_win_rate": round(black_wins / total_games, 4) if total_games > 0 else 0.0,
            "white_win_rate": round(white_wins / total_games, 4) if total_games > 0 else 0.0,
        },
        "stats": {
            "total_raw_states": total_raw,
            "total_unique_states": total_unique,
            "total_duplicates_removed": total_duplicates,
            "total_leakage_with_train_removed": total_leakage,
            "deduplication_ratio": round(dedup_ratio, 4),
            "splits": {
                "train": {
                    "num_games": len(train_games),
                    **train_stats,
                },
                "val": {
                    "num_games": len(val_games),
                    **val_stats,
                },
                "test": {
                    "num_games": len(test_games),
                    **test_stats,
                },
            },
        },
        "output_files": {
            "unified": str(unified_path),
            "train": str(output_dir / f"{dataset_name}_train.npz"),
            "val": str(output_dir / f"{dataset_name}_val.npz"),
            "test": str(output_dir / f"{dataset_name}_test.npz"),
            "parts_dir": str(parts_dir),
        },
    }

    log_path = output_dir / f"{dataset_name}_log.json"
    log_path.write_text(json.dumps(log_info, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"------------------------------------------------------------")
        print(f"Hoàn thành trong: {total_duration:.2f}s ({total_duration/total_games:.3f}s/ván)")
        print(f"Kết quả ván: Đen thắng {black_wins}, Trắng thắng {white_wins}, Hoà {draws}")
        print(f"Tổng số thế cờ ban đầu: {total_raw}")
        print(f"Tổng số thế cờ duy nhất (sau 8 đối xứng): {total_unique}")
        print(f"Số bản trùng nội bộ đã gộp/lấy TB: {total_duplicates} ({dedup_ratio*100:.1f}%)")
        print(f"Số thế cờ trùng Train đã loại bỏ khỏi Val/Test (chống rò rỉ): {total_leakage}")
        print(f"Chia tập: Train {train_stats['unique_count']} ({len(train_games)} ván) | "
              f"Val {val_stats['unique_count']} ({len(val_games)} ván) | "
              f"Test {test_stats['unique_count']} ({len(test_games)} ván)")
        print(f"Đã lưu:")
        print(f"  - Dataset: {unified_path}")
        print(f"  - Parts:   {parts_dir}")
        print(f"  - Log:     {log_path}")
        print(f"============================================================")

    return log_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu huấn luyện Supervised Learning cho Caro 11x11 từ Alpha-Beta."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG_PATH,
        help=f"Đường dẫn file cấu hình YAML (mặc định: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Chỉ sinh tối đa N ván (dùng cho test/debug)")
    parser.add_argument("--workers", type=int, default=None, help="Số tiến trình chạy song song")
    parser.add_argument("--chunk-size", type=int, default=None, help="Kích thước mỗi chunk checkpoint (mặc định 500)")
    parser.add_argument("--no-resume", action="store_true", help="Không tải lại các part file đã có, chạy lại từ đầu")
    parser.add_argument("--out-dir", default=None, help="Thư mục xuất dữ liệu (mặc định trong config)")
    parser.add_argument("--name", default=None, help="Tên bộ dữ liệu (mặc định trong config)")
    args = parser.parse_args()

    cfg = load_generation_config(args.config)
    generate_dataset(
        cfg=cfg,
        limit=args.limit,
        workers=args.workers,
        out_dir=args.out_dir,
        name=args.name,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

