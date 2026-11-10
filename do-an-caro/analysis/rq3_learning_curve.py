"""Phân tích và vẽ đường cong học tập cho RQ3: SL vs RL self-play (Issue #11).

Đọc log huấn luyện (`checkpoints/level6/train_log.jsonl`) và/hoặc log giải đấu
chính thức (`logs/rq3_rl_vs_sl.jsonl`) để:
1. Tạo bảng Markdown so sánh các checkpoint RL self-play với mạng Cấp 5 (SL),
   kèm khoảng tin cậy Wilson 95% và phân tách theo màu quân X/O.
2. Vẽ biểu đồ đường cong học tập (`analysis/figures/rq3_learning_curve.png`).

Chạy:
  python -m analysis.rq3_learning_curve \\
      --train-log checkpoints/level6/train_log.jsonl \\
      --arena-log logs/rq3_rl_vs_sl.jsonl \\
      --fig analysis/figures/rq3_learning_curve.png \\
      --out analysis/rq3_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Tránh xung đột libiomp5md.dll giữa MKL (Anaconda) và PyTorch khi vẽ biểu đồ trên Windows
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.arena_stats import agents_in, color_table, load_games, pair_table


def load_train_log(path: str | Path) -> list[dict[str, Any]]:
    """Đọc các dòng JSON từ `train_log.jsonl` của `train_rl`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy train log: {p}")
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return sorted(records, key=lambda r: int(r.get("round", 0)))


def summarize_arena_rq3(arena_log: str | Path, baseline_label: str = "sl") -> list[dict[str, Any]]:
    """Tổng hợp kết quả đối đầu giữa các checkpoint RL và SL từ log arena."""
    games = load_games(str(arena_log))
    pt = pair_table(games)
    ct = color_table(games)
    labels = [lbl for lbl in agents_in(games) if lbl != baseline_label]

    rows: list[dict[str, Any]] = []
    for lbl in labels:
        row = pt.get((lbl, baseline_label))
        if row is None:
            continue
        c_info = ct["agents"].get(lbl, {})
        rows.append({
            "label": lbl,
            "n": row["n"],
            "wins": row["wins"],
            "draws": row["draws"],
            "losses": row["losses"],
            "score": row["score"],
            "ci95": row["ci95"],
            "score_as_x": c_info.get("score_as_x"),
            "score_as_o": c_info.get("score_as_o"),
        })
    return rows


def build_rq3_markdown(
    train_records: list[dict[str, Any]] | None,
    arena_rows: list[dict[str, Any]] | None,
) -> str:
    """Tạo báo cáo Markdown đầy đủ cho RQ3."""
    lines = ["# Kết quả thí nghiệm RQ3: SL vs RL Self-Play (Cấp 6)", ""]

    if arena_rows:
        lines.extend([
            "## 1. Đối đầu chính thức trên Arena (SL làm chuẩn đối chiếu)",
            "",
            "| Checkpoint RL | Số ván | Thắng | Hòa | Thua | Score vs SL | Wilson 95% | Score (X) | Score (O) |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for r in arena_rows:
            lo, hi = r["ci95"]
            sx = f"{r['score_as_x']:.3f}" if r["score_as_x"] is not None else "–"
            so = f"{r['score_as_o']:.3f}" if r["score_as_o"] is not None else "–"
            lines.append(
                f"| **{r['label']}** | {r['n']} | {r['wins']} | {r['draws']} | {r['losses']} "
                f"| {r['score']:.3f} | [{lo:.3f}, {hi:.3f}] | {sx} | {so} |"
            )
        lines.append("")

    if train_records:
        lines.extend([
            "## 2. Đường cong học tập theo vòng self-play (`train_log.jsonl`)",
            "",
            "| Vòng | Episodes | Policy Loss | Value Loss | Entropy | Mean Reward | Score vs SL | Wilson 95% | Best |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for rec in train_records:
            lo, hi = rec.get("ci95_vs_sl", [0.0, 0.0])
            pl = f"{rec['policy_loss']:.4f}" if rec.get("policy_loss") is not None else "–"
            vl = f"{rec['value_loss']:.4f}" if rec.get("value_loss") is not None else "–"
            ent = f"{rec['entropy']:.3f}" if rec.get("entropy") is not None else "–"
            best_mark = "✓" if rec.get("is_best") else ""
            lines.append(
                f"| {rec['round']} | {rec['episodes']:,} | {pl} | {vl} | {ent} "
                f"| {rec.get('mean_reward', 0.0):+.3f} | {rec['score_vs_sl']:.3f} | [{lo:.3f}, {hi:.3f}] | {best_mark} |"
            )
        lines.append("")

    return "\n".join(lines)


def plot_rq3_learning_curve(
    out_fig: str | Path,
    train_records: list[dict[str, Any]] | None = None,
    arena_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Vẽ biểu đồ đường cong học tập RQ3 (score vs SL + chỉ số Actor-Critic) và lưu file PNG."""
    out_path = Path(out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_train = bool(train_records)
    has_arena = bool(arena_rows)
    if not has_train and not has_arena:
        raise ValueError("Cần ít nhất một nguồn dữ liệu (`train_records` hoặc `arena_rows`).")

    ncols = 2 if (has_train and has_arena) or has_train else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    ax_idx = 0
    if has_train and train_records is not None:
        eps = [r["episodes"] for r in train_records]
        scores = [r["score_vs_sl"] * 100.0 for r in train_records]
        ci_lo = [r["ci95_vs_sl"][0] * 100.0 for r in train_records]
        ci_hi = [r["ci95_vs_sl"][1] * 100.0 for r in train_records]

        ax0 = axes[ax_idx]
        ax_idx += 1
        ax0.plot(eps, scores, "o-", color="#1f77b4", linewidth=2, label="RL Self-Play vs SL")
        ax0.fill_between(eps, ci_lo, ci_hi, color="#1f77b4", alpha=0.18, label="Wilson 95% CI")
        ax0.axhline(50.0, color="#d62728", linestyle="--", linewidth=1.2, label="Mốc hòa SL (50%)")
        ax0.set_title("Đường cong học tập RQ3: Tỷ lệ thắng RL vs SL")
        ax0.set_xlabel("Số ván Self-Play (Episodes)")
        ax0.set_ylabel("Score vs SL (%)")
        ax0.set_ylim(0, 105)
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best")

        if not has_arena:
            # Đồ thị phụ: Value loss và Entropy qua các vòng cập nhật
            upd = [r for r in train_records if r.get("value_loss") is not None]
            if upd:
                u_eps = [r["episodes"] for r in upd]
                v_loss = [r["value_loss"] for r in upd]
                ent = [r["entropy"] for r in upd]
                ax1 = axes[ax_idx]
                l1 = ax1.plot(u_eps, v_loss, "s-", color="#ff7f0e", label="Value MSE Loss")
                ax1.set_xlabel("Số ván Self-Play (Episodes)")
                ax1.set_ylabel("Value Loss", color="#ff7f0e")
                ax1.grid(True, alpha=0.3)
                ax1_r = ax1.twinx()
                l2 = ax1_r.plot(u_eps, ent, "^--", color="#2ca02c", label="Policy Entropy")
                ax1_r.set_ylabel("Entropy (nats)", color="#2ca02c")
                lines = l1 + l2
                ax1.legend(lines, [l.get_label() for l in lines], loc="best")
                ax1.set_title("Động lực học Actor-Critic (Value Loss & Entropy)")

    if has_arena and arena_rows is not None:
        ax = axes[ax_idx]
        labels = [r["label"] for r in arena_rows]
        x = list(range(len(labels)))
        scores = [r["score"] * 100.0 for r in arena_rows]
        yerr_lo = [max(0.0, (r["score"] - r["ci95"][0]) * 100.0) for r in arena_rows]
        yerr_hi = [max(0.0, (r["ci95"][1] - r["score"]) * 100.0) for r in arena_rows]

        ax.errorbar(x, scores, yerr=[yerr_lo, yerr_hi], fmt="o-", capsize=5, color="#1f77b4", linewidth=2, label="Arena Score vs SL")
        ax.axhline(50.0, color="#d62728", linestyle="--", linewidth=1.2, label="Mốc hòa SL (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Score vs SL (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Đối đầu Arena RQ3: Các checkpoint RL vs SL (200 ván/cặp)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích và vẽ đường cong học tập RQ3 (Issue #11).")
    parser.add_argument("--train-log", help="Đường dẫn tới checkpoints/level6/train_log.jsonl")
    parser.add_argument("--arena-log", help="Đường dẫn tới logs/rq3_rl_vs_sl.jsonl")
    parser.add_argument("--baseline", default="sl", help="Nhãn tác tử SL chuẩn trong arena log (mặc định: sl)")
    parser.add_argument("--fig", default="analysis/figures/rq3_learning_curve.png", help="Đường dẫn file ảnh PNG đầu ra.")
    parser.add_argument("--out", help="Đường dẫn lưu báo cáo Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_log and not args.arena_log:
        print("Lỗi: Cần truyền ít nhất `--train-log` hoặc `--arena-log`.", file=sys.stderr)
        sys.exit(1)

    train_records = load_train_log(args.train_log) if args.train_log else None
    arena_rows = summarize_arena_rq3(args.arena_log, args.baseline) if args.arena_log else None

    report = build_rq3_markdown(train_records, arena_rows)
    print(report)
    if args.out:
        out_md = Path(args.out)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")

    fig_path = plot_rq3_learning_curve(args.fig, train_records=train_records, arena_rows=arena_rows)
    print(f"Đã lưu biểu đồ đường cong học tập vào: {fig_path}")


if __name__ == "__main__":
    main()
