"""Thống kê từ log arena (proposal mục 6, đề mục 3.5).

- score = (thắng + 0,5 × hòa) / tổng số ván
- Ma trận đối đầu: score của hàng gặp cột, kèm khoảng tin cậy Wilson 95%
- Score tách theo màu quân (kiểm tra lợi thế đi trước)
- Thời gian trung bình mỗi nước, độ trễ p95, số nút / số mô phỏng trung bình
- Xếp hạng Bradley-Terry quy ra Elo, khoảng tin cậy bằng bootstrap theo ván

Ghi chú: Wilson áp cho score (hòa tính 0,5) là xấp xỉ thường dùng khi tỷ lệ hòa thấp.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from arena.logger import load_records

Z95 = 1.959963984540054
BT_MAX_ITERS = 2000
BT_TOL = 1e-10
BT_PRIOR_GAMES = 1.0     # mỗi tác tử thêm 1 ván hòa ảo với đối thủ chuẩn, tránh Elo vô cực
ELO_SCALE = 400.0
DEFAULT_BOOTSTRAP = 1000
COST_KEYS = ("nodes", "sims", "depth_reached")


def load_games(path: str) -> list[dict[str, Any]]:
    """Đọc log JSONL của arena."""
    return load_records(path)


def game_points(game: dict[str, Any]) -> tuple[str, str, float]:
    """(nhãn X, nhãn O, điểm của X: 1 thắng / 0,5 hòa / 0 thua)."""
    x = game["agent_x"]["label"]
    o = game["agent_o"]["label"]
    result = game["result"]
    return x, o, 1.0 if result == 1 else 0.5 if result == 0 else 0.0


def wilson_interval(score: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỷ lệ `score` trên `n` ván."""
    if n == 0:
        return (math.nan, math.nan)
    denom = 1 + z * z / n
    center = (score + z * z / (2 * n)) / denom
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def agents_in(games: Iterable[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for g in games:
        for side in ("agent_x", "agent_o"):
            label = g[side]["label"]
            if label not in labels:
                labels.append(label)
    return labels


def pair_table(games: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Kết quả từng cặp có thứ tự (a, b) theo góc nhìn a."""
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0, "w": 0, "d": 0, "l": 0})
    for g in games:
        x, o, px = game_points(g)
        for me, opp, pts in ((x, o, px), (o, x, 1.0 - px)):
            row = acc[(me, opp)]
            row["n"] += 1
            if pts == 1.0:
                row["w"] += 1
            elif pts == 0.5:
                row["d"] += 1
            else:
                row["l"] += 1
    table = {}
    for key, row in acc.items():
        n = int(row["n"])
        score = (row["w"] + 0.5 * row["d"]) / n
        table[key] = {
            "n": n, "wins": int(row["w"]), "draws": int(row["d"]), "losses": int(row["l"]),
            "win_rate": row["w"] / n, "draw_rate": row["d"] / n, "loss_rate": row["l"] / n,
            "score": score, "ci95": wilson_interval(score, n),
        }
    return table


def color_table(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Score mỗi tác tử khi cầm X (đi trước) và khi cầm O, cùng score chung của bên X."""
    per = defaultdict(lambda: {"x_n": 0, "x_pts": 0.0, "o_n": 0, "o_pts": 0.0})
    x_points = 0.0
    for g in games:
        x, o, px = game_points(g)
        per[x]["x_n"] += 1
        per[x]["x_pts"] += px
        per[o]["o_n"] += 1
        per[o]["o_pts"] += 1.0 - px
        x_points += px
    agents = {
        label: {
            "score_as_x": v["x_pts"] / v["x_n"] if v["x_n"] else math.nan,
            "score_as_o": v["o_pts"] / v["o_n"] if v["o_n"] else math.nan,
            "games_as_x": v["x_n"],
            "games_as_o": v["o_n"],
        }
        for label, v in per.items()
    }
    n = len(games)
    x_score = x_points / n if n else math.nan
    return {"agents": agents, "x_score": x_score, "x_ci95": wilson_interval(x_score, n), "games": n}


def timing_table(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Thời gian mỗi nước (trung bình, p95) và chi phí tìm kiếm trung bình theo tác tử."""
    times: dict[str, list[float]] = defaultdict(list)
    costs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for g in games:
        labels = {1: g["agent_x"]["label"], -1: g["agent_o"]["label"]}
        for mv in g["moves"]:
            label = labels[mv["player"]]
            times[label].append(mv["time_ms"])
            for key in COST_KEYS:
                value = (mv.get("stats") or {}).get(key)
                if isinstance(value, (int, float)):
                    costs[label][key].append(value)
    out = {}
    for label, ts in times.items():
        arr = np.asarray(ts, dtype=float)
        row: dict[str, Any] = {
            "moves": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }
        for key in COST_KEYS:
            vals = costs[label].get(key)
            row[f"mean_{key}"] = float(np.mean(vals)) if vals else None
        out[label] = row
    return out


def bradley_terry(games: list[dict[str, Any]], labels: list[str] | None = None) -> dict[str, float]:
    """Độ mạnh Bradley-Terry bằng thuật toán MM (Hunter, 2004); hòa tính 0,5 thắng mỗi bên.

    Mỗi tác tử được thêm BT_PRIOR_GAMES ván hòa ảo với một đối thủ chuẩn độ mạnh 1,
    để tác tử thắng/thua tất cả vẫn có độ mạnh hữu hạn. Trả về Elo, trung bình bằng 0.
    """
    labels = labels or agents_in(games)
    idx = {a: i for i, a in enumerate(labels)}
    k = len(labels)
    wins = np.zeros((k, k))
    for g in games:
        x, o, px = game_points(g)
        wins[idx[x], idx[o]] += px
        wins[idx[o], idx[x]] += 1.0 - px
    n_games = wins + wins.T
    total_wins = wins.sum(axis=1) + 0.5 * BT_PRIOR_GAMES
    p = np.ones(k)
    for _ in range(BT_MAX_ITERS):
        denom = (n_games / (p[:, None] + p[None, :])).sum(axis=1) + BT_PRIOR_GAMES / (p + 1.0)
        new_p = total_wins / denom
        if np.max(np.abs(new_p - p)) < BT_TOL:
            p = new_p
            break
        p = new_p
    elo = ELO_SCALE * np.log10(p)
    elo -= elo.mean()
    return {a: float(elo[idx[a]]) for a in labels}


def bootstrap_elo(
    games: list[dict[str, Any]],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Elo Bradley-Terry kèm khoảng tin cậy 95% bằng bootstrap theo ván."""
    labels = agents_in(games)
    point = bradley_terry(games, labels)
    rng = np.random.default_rng(seed)
    samples = {a: [] for a in labels}
    n = len(games)
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        elo = bradley_terry([games[i] for i in pick], labels)
        for a in labels:
            samples[a].append(elo[a])
    return {
        a: {
            "elo": point[a],
            "ci95": (float(np.percentile(samples[a], 2.5)), float(np.percentile(samples[a], 97.5))),
        }
        for a in labels
    }


def error_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Các ván kết thúc vì tác tử lỗi hoặc đi nước bất hợp lệ."""
    return [g for g in games if g.get("error")]
