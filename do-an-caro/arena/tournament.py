"""Giải đấu round-robin theo file cấu hình YAML, chạy đa tiến trình, ghi log JSONL.

Mỗi cặp tác tử đấu trên N khai cuộc cố định; mỗi khai cuộc chơi 2 ván đổi bên
đi trước (100 khai cuộc -> 200 ván/cặp). Cặp có tác tử ngẫu nhiên (Random, MCTS...)
chạy thêm theo mọi seed trong `seeds`; cặp toàn tác tử tất định chỉ chạy seed đầu
vì chạy lại sẽ ra đúng các ván cũ.

Hai chế độ (proposal mục 6):
  A – ngân sách cố định: mọi tác tử nhận `budget` giây mỗi nước.
  B – cấu hình mặc định minh bạch: budget=None, mỗi tác tử dùng cấu hình chuẩn.

Log ghi nối tiếp; chạy lại cùng cấu hình sẽ bỏ qua các ván đã có (chạy tiếp khi bị ngắt).

Chạy:  python -m arena.tournament configs/arena/smoke.yaml [--workers 4] [--limit 20]
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from arena.agent_factory import AgentSpec
from arena.logger import JsonlLogger, completed_game_ids, write_run_meta
from arena.match import GameTask, play_game
from arena.openings import load_openings
from game.constants import DEFAULT_BOARD_SIZE, ENGINE_VERSION, RULE_VIETNAM

MODES = ("A", "B")
DEFAULT_LOG_DIR = "logs"
PROGRESS_EVERY = 10
SEED_STRIDE = 1_000_003  # tách không gian seed giữa các lần chạy seed khác nhau


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mode = str(cfg.get("mode", "A")).upper()
    if mode not in MODES:
        raise ValueError(f"mode phải là A hoặc B, nhận '{mode}'.")
    cfg["mode"] = mode
    if mode == "A" and cfg.get("budget") is None:
        raise ValueError("Chế độ A cần `budget` (giây/nước).")
    if mode == "B":
        cfg["budget"] = None
    if len(cfg.get("agents") or {}) < 2:
        raise ValueError("Cần ít nhất 2 tác tử.")
    return cfg


def agent_seed(run_seed: int, opening_id: int, swap: int, side: int) -> int:
    """Seed riêng, cố định cho từng tác tử trong từng ván (tái lập được)."""
    return run_seed * SEED_STRIDE + opening_id * 4 + swap * 2 + side


def build_schedule(cfg: dict[str, Any], openings: list[dict[str, Any]]) -> list[GameTask]:
    """Lập danh sách mọi ván cần chơi theo cấu hình."""
    specs = [AgentSpec.from_config(label, a) for label, a in cfg["agents"].items()]
    pairs = cfg.get("pairs")
    if pairs:
        by_label = {s.label: s for s in specs}
        pair_specs = [(by_label[a], by_label[b]) for a, b in pairs]
    else:
        pair_specs = list(itertools.combinations(specs, 2))

    num_openings = int(cfg.get("num_openings", len(openings)))
    if num_openings > len(openings):
        raise ValueError(f"Cần {num_openings} khai cuộc nhưng file chỉ có {len(openings)}.")
    chosen = openings[:num_openings]
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    size = int(cfg.get("size", DEFAULT_BOARD_SIZE))
    rule = str(cfg.get("rule", RULE_VIETNAM))

    tasks: list[GameTask] = []
    for a, b in pair_specs:
        pair_seeds = seeds if (a.stochastic or b.stochastic) else seeds[:1]
        for run_seed in pair_seeds:
            for op in chosen:
                for swap in (0, 1):
                    x, o = (a, b) if swap == 0 else (b, a)
                    tasks.append(GameTask(
                        game_id=f"{a.label}__{b.label}__s{run_seed}__o{op['id']:03d}__{'ab' if swap == 0 else 'ba'}",
                        experiment=cfg["name"],
                        mode=cfg["mode"],
                        budget=cfg.get("budget"),
                        size=size,
                        rule=rule,
                        opening_id=int(op["id"]),
                        opening_moves=tuple(int(m) for m in op["moves"]),
                        seed=run_seed,
                        agent_x=x,
                        agent_o=o,
                        seed_x=agent_seed(run_seed, int(op["id"]), swap, 0),
                        seed_o=agent_seed(run_seed, int(op["id"]), swap, 1),
                    ))
    return tasks


def _limit_threads() -> None:
    """Mỗi tiến trình dùng 1 luồng CPU (chế độ A: 1 luồng/tác tử, proposal mục 6)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def run_tournament(
    cfg: dict[str, Any],
    log_dir: str | Path = DEFAULT_LOG_DIR,
    workers: int | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> Path:
    """Chơi các ván còn thiếu và ghi vào <log_dir>/<name>.jsonl. Trả về đường dẫn log."""
    openings = load_openings(cfg["openings_file"])
    tasks = build_schedule(cfg, openings)
    log_path = Path(log_dir) / f"{cfg['name']}.jsonl"
    done = completed_game_ids(log_path)
    pending = [t for t in tasks if t.game_id not in done]
    if limit is not None:
        pending = pending[:limit]
    workers = int(workers or cfg.get("workers", 1))

    write_run_meta(log_path, {
        "experiment": cfg["name"],
        "engine_version": ENGINE_VERSION,
        "config": cfg,
        "workers": workers,
        "games_total": len(tasks),
        "games_already_done": len(done),
        "games_this_run": len(pending),
    })
    logger = JsonlLogger(log_path)
    if verbose:
        print(f"[{cfg['name']}] {len(tasks)} ván, đã có {len(done)}, chạy {len(pending)} ván "
              f"với {workers} tiến trình -> {log_path}")

    started = time.perf_counter()
    finished = 0

    def report(record: dict[str, Any]) -> None:
        nonlocal finished
        logger.write(record)
        finished += 1
        if verbose and (finished % PROGRESS_EVERY == 0 or finished == len(pending)):
            print(f"  {finished}/{len(pending)} ván, {time.perf_counter() - started:.0f}s")

    if workers <= 1:
        for task in pending:
            report(play_game(task))
    else:
        _limit_threads()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(play_game, t) for t in pending]
            for fut in as_completed(futures):
                report(fut.result())
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy giải đấu round-robin theo file YAML.")
    parser.add_argument("config", help="Đường dẫn file cấu hình, ví dụ configs/arena/smoke.yaml")
    parser.add_argument("--workers", type=int, default=None, help="Số tiến trình (mặc định lấy từ YAML)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy tối đa N ván còn thiếu")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_tournament(cfg, log_dir=args.log_dir, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
