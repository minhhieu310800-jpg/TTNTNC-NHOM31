"""In bảng tổng hợp (Markdown) từ log arena.

Chạy:  python -m analysis.summarize logs/smoke.jsonl [--bootstrap 1000] [--out bao_cao.md]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from analysis.arena_stats import (
    DEFAULT_BOOTSTRAP,
    agents_in,
    bootstrap_elo,
    color_table,
    error_games,
    load_games,
    pair_table,
    timing_table,
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def matrix_markdown(games: list[dict[str, Any]]) -> str:
    labels = agents_in(games)
    table = pair_table(games)
    lines = ["| hàng gặp cột | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for a in labels:
        cells = []
        for b in labels:
            row = table.get((a, b))
            if a == b or row is None:
                cells.append("–")
            else:
                lo, hi = row["ci95"]
                cells.append(f"{row['score']:.2f} [{lo:.2f}, {hi:.2f}] (n={row['n']})")
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pairs_markdown(games: list[dict[str, Any]]) -> str:
    table = pair_table(games)
    labels = agents_in(games)
    lines = ["| A | B | n | thắng | hòa | thua | score A | Wilson 95% |", "|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            row = table.get((a, b))
            if row is None:
                continue
            lo, hi = row["ci95"]
            lines.append(f"| {a} | {b} | {row['n']} | {row['wins']} | {row['draws']} | {row['losses']} "
                         f"| {row['score']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    return "\n".join(lines)


def color_markdown(games: list[dict[str, Any]]) -> str:
    ct = color_table(games)
    lo, hi = ct["x_ci95"]
    lines = [f"Score chung của bên X (đi trước): **{ct['x_score']:.3f}** [{lo:.3f}, {hi:.3f}] trên {ct['games']} ván.",
             "", "| tác tử | score khi cầm X | score khi cầm O | ván X | ván O |", "|---|---|---|---|---|"]
    for label, v in ct["agents"].items():
        lines.append(f"| {label} | {_fmt(v['score_as_x'])} | {_fmt(v['score_as_o'])} "
                     f"| {v['games_as_x']} | {v['games_as_o']} |")
    return "\n".join(lines)


def timing_markdown(games: list[dict[str, Any]]) -> str:
    tt = timing_table(games)
    lines = ["| tác tử | số nước | TB (ms) | p95 (ms) | max (ms) | TB nodes | TB sims | TB độ sâu |",
             "|---|---|---|---|---|---|---|---|"]
    for label, v in tt.items():
        lines.append(f"| {label} | {v['moves']} | {_fmt(v['mean_ms'], 1)} | {_fmt(v['p95_ms'], 1)} "
                     f"| {_fmt(v['max_ms'], 1)} | {_fmt(v['mean_nodes'], 0)} | {_fmt(v['mean_sims'], 0)} "
                     f"| {_fmt(v['mean_depth_reached'], 2)} |")
    return "\n".join(lines)


def elo_markdown(games: list[dict[str, Any]], n_boot: int) -> str:
    elo = bootstrap_elo(games, n_boot=n_boot)
    lines = [f"Bradley-Terry quy ra Elo (trung bình = 0), khoảng tin cậy 95% bootstrap {n_boot} lần.",
             "", "| hạng | tác tử | Elo | 95% CI |", "|---|---|---|---|"]
    for rank, (label, v) in enumerate(sorted(elo.items(), key=lambda kv: -kv[1]["elo"]), start=1):
        lo, hi = v["ci95"]
        lines.append(f"| {rank} | {label} | {v['elo']:.0f} | [{lo:.0f}, {hi:.0f}] |")
    return "\n".join(lines)


def summarize(path: str, n_boot: int = DEFAULT_BOOTSTRAP) -> str:
    games = load_games(path)
    if not games:
        return f"Log {path} chưa có ván nào."
    errors = error_games(games)
    parts = [
        f"# Tổng hợp giải đấu: {Path(path).stem}",
        f"Số ván: {len(games)}. Ván kết thúc do lỗi tác tử: {len(errors)}.",
        "## Ma trận đối đầu (score của hàng gặp cột, Wilson 95%)", matrix_markdown(games),
        "## Kết quả từng cặp", pairs_markdown(games),
        "## Lợi thế đi trước", color_markdown(games),
        "## Thời gian và chi phí tìm kiếm", timing_markdown(games),
        "## Xếp hạng", elo_markdown(games, n_boot),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp log arena thành bảng Markdown.")
    parser.add_argument("log", help="File log JSONL, ví dụ logs/smoke.jsonl")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--out", default=None, help="Ghi ra file Markdown thay vì in ra màn hình")
    args = parser.parse_args()
    text = summarize(args.log, args.bootstrap)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Đã ghi {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
