"""Sinh bộ khai cuộc cân bằng dùng chung cho mọi trận (proposal mục 6).

Mỗi khai cuộc gồm 2–4 nước ngẫu nhiên gần tâm (seed cố định). Thế cờ được
Alpha-Beta (độ sâu nhỏ) chấm điểm từ góc nhìn người đến lượt; chỉ giữ khai cuộc
có |điểm| không vượt ngưỡng, tức không bên nào lợi rõ rệt. Khai cuộc trùng nhau
qua 8 phép đối xứng của bàn vuông bị loại.

Chạy:  python -m arena.openings --n 100 --seed 2026 --out configs/openings.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from agents import AlphaBetaAgent
from game.constants import DEFAULT_BOARD_SIZE, ENGINE_VERSION, RULE_VIETNAM
from game.game import CaroGame

DEFAULT_N = 100
DEFAULT_SEED = 2026
MIN_MOVES = 2
MAX_MOVES = 4
CENTER_RADIUS = 2          # nước khai cuộc nằm trong ô vuông 5×5 quanh tâm
EVAL_DEPTH = 2             # độ sâu Alpha-Beta dùng để chấm khai cuộc
BALANCE_THRESHOLD = 300.0  # |điểm| tối đa để coi là cân bằng (thang fast_evaluate)
MAX_TRIES_PER_OPENING = 200
DEFAULT_OUT = "configs/openings.json"


def canonical_key(board: np.ndarray) -> bytes:
    """Khóa nhỏ nhất qua 8 phép đối xứng (xoay 0/90/180/270 độ, có/không lật)."""
    variants = []
    for k in range(4):
        rotated = np.rot90(board, k)
        variants.append(rotated.tobytes())
        variants.append(np.fliplr(rotated).tobytes())
    return min(variants)


def evaluate_opening(game: CaroGame, depth: int = EVAL_DEPTH) -> float:
    """Điểm Alpha-Beta của thế cờ theo góc nhìn người đến lượt (dương = người đến lượt hơn)."""
    agent = AlphaBetaAgent(depth=depth, fast_eval=True)
    agent.select_action(game.clone())
    return float(agent.last_stats["score"])


def generate_openings(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_BOARD_SIZE,
    rule: str = RULE_VIETNAM,
    min_moves: int = MIN_MOVES,
    max_moves: int = MAX_MOVES,
    radius: int = CENTER_RADIUS,
    depth: int = EVAL_DEPTH,
    threshold: float = BALANCE_THRESHOLD,
) -> dict[str, Any]:
    """Trả về {"meta": ..., "openings": [{"id", "moves", "eval"}, ...]}."""
    rng = random.Random(seed)
    center = size // 2
    cells = [
        r * size + c
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
    ]
    openings: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    tries = 0
    rejected_unbalanced = 0
    while len(openings) < n:
        tries += 1
        if tries > n * MAX_TRIES_PER_OPENING:
            raise RuntimeError(
                f"Chỉ sinh được {len(openings)}/{n} khai cuộc; hãy nới ngưỡng cân bằng."
            )
        k = rng.randint(min_moves, max_moves)
        moves = rng.sample(cells, k)
        game = CaroGame(size=size, rule=rule)
        for action in moves:
            game.step(action)
        key = canonical_key(game.board)
        if key in seen:
            continue
        seen.add(key)
        score = evaluate_opening(game, depth)
        if abs(score) > threshold:
            rejected_unbalanced += 1
            continue
        openings.append({"id": len(openings), "moves": moves, "eval": score})

    meta = {
        "n": n,
        "seed": seed,
        "size": size,
        "rule": rule,
        "moves_range": [min_moves, max_moves],
        "center_radius": radius,
        "evaluator": f"AlphaBetaAgent(depth={depth}, fast_eval=True)",
        "balance_threshold": threshold,
        "engine_version": ENGINE_VERSION,
        "tries": tries,
        "rejected_unbalanced": rejected_unbalanced,
    }
    return {"meta": meta, "openings": openings}


def save_openings(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_openings(path: str | Path) -> list[dict[str, Any]]:
    """Đọc danh sách khai cuộc từ file JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["openings"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh bộ khai cuộc cân bằng cho arena.")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=BALANCE_THRESHOLD)
    parser.add_argument("--depth", type=int, default=EVAL_DEPTH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    data = generate_openings(n=args.n, seed=args.seed, depth=args.depth, threshold=args.threshold)
    save_openings(data, args.out)
    meta = data["meta"]
    print(f"Đã lưu {meta['n']} khai cuộc vào {args.out} "
          f"(thử {meta['tries']}, loại vì lệch {meta['rejected_unbalanced']}).")


if __name__ == "__main__":
    main()
