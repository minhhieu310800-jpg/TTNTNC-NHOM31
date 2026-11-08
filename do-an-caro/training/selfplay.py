"""Self-play bằng PUCT cho cấp 8 và các phép đối xứng để tăng dữ liệu.

Nhiều ván chạy cùng lúc: ở mỗi lượt mô phỏng, lá của tất cả các ván được gom
thành một batch gửi cho mạng (hiệu quả trên GPU, proposal mục 5).
Mỗi thế cờ sinh một mẫu (planes, pi, z): pi là phân bố số lượt thăm ở gốc,
z là kết quả cuối ván theo góc nhìn người đến lượt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.network import encode_game
from agents.puct_agent import DEFAULT_C_PUCT, DEFAULT_FPU, DEFAULT_RADIUS, Evaluator, PUCTTree
from game.constants import DEFAULT_BOARD_SIZE, RULE_VIETNAM
from game.game import CaroGame

NUM_SYMMETRIES = 8


@dataclass
class SelfPlayConfig:
    simulations: int = 100
    c_puct: float = DEFAULT_C_PUCT
    dirichlet_alpha: float = 0.15   # ~10 / số nước hợp lệ trung bình
    dirichlet_eps: float = 0.25
    temp_moves: int = 10            # số nước đầu chọn theo phân bố lượt thăm (khám phá)
    radius: int | None = DEFAULT_RADIUS   # chỉ xét ô trong bán kính quanh quân đã đánh
    fpu: float = DEFAULT_FPU
    size: int = DEFAULT_BOARD_SIZE
    rule: str = RULE_VIETNAM


def symmetry(planes: np.ndarray, pi: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Phép đối xứng thứ k (0..7): xoay k%4 lần 90 độ, lật ngang nếu k >= 4.

    `planes` có dạng (..., C, S, S); `pi` có dạng (..., S*S).
    """
    size = planes.shape[-1]
    p = np.rot90(planes, k % 4, axes=(-2, -1))
    q = np.rot90(pi.reshape(*pi.shape[:-1], size, size), k % 4, axes=(-2, -1))
    if k >= 4:
        p = np.flip(p, axis=-1)
        q = np.flip(q, axis=-1)
    return np.ascontiguousarray(p), np.ascontiguousarray(q).reshape(pi.shape)


def play_games(
    evaluator: Evaluator,
    n_games: int,
    cfg: SelfPlayConfig,
    seed: int,
) -> dict[str, Any]:
    """Chơi `n_games` ván self-play song song; trả về mẫu huấn luyện và thống kê ván."""
    rng = np.random.default_rng(seed)
    num_cells = cfg.size * cfg.size
    games = [CaroGame(size=cfg.size, rule=cfg.rule) for _ in range(n_games)]
    trees = [
        PUCTTree(g.current_player, cfg.c_puct, (cfg.dirichlet_alpha, cfg.dirichlet_eps), rng,
                 radius=cfg.radius, fpu=cfg.fpu)
        for g in games
    ]
    records: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active = list(range(n_games))

    while active:
        for _ in range(cfg.simulations):
            leaves = [trees[i].select_leaf(games[i]) for i in active]
            pending = [j for j, leaf in enumerate(leaves) if leaf.value is None]
            if pending:
                planes = np.stack([leaves[j].planes for j in pending])
                masks = np.stack([leaves[j].mask for j in pending])
                priors, values = evaluator.evaluate(planes, masks)
            out = {j: k for k, j in enumerate(pending)}
            for j, i in enumerate(active):
                if j in out:
                    trees[i].backup(leaves[j], priors[out[j]], values[out[j]])
                else:
                    trees[i].backup(leaves[j])

        for i in active:
            game, tree = games[i], trees[i]
            pi = tree.visit_policy(num_cells)
            planes, _ = encode_game(game)
            records[i].append((planes.astype(np.uint8), pi, game.current_player))
            if game.move_count < cfg.temp_moves:
                action = int(rng.choice(num_cells, p=pi / pi.sum()))
            else:
                action = int(np.argmax(pi))
            game.step(action)
            tree.advance(action)
        active = [i for i in active if not games[i].is_terminal()]

    planes_out, pi_out, z_out = [], [], []
    results, lengths = [], []
    for game, recs in zip(games, records):
        result = int(game.result())
        results.append(result)
        lengths.append(game.move_count)
        for planes, pi, to_play in recs:
            planes_out.append(planes)
            pi_out.append(pi)
            z_out.append(0.0 if result == 0 else (1.0 if result == to_play else -1.0))
    return {
        "planes": np.stack(planes_out),
        "pi": np.stack(pi_out).astype(np.float32),
        "z": np.asarray(z_out, dtype=np.float32),
        "results": np.asarray(results, dtype=np.int8),
        "lengths": np.asarray(lengths, dtype=np.int16),
    }
