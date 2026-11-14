"""Sinh toàn bộ hình cho Chương 4 từ log trong results/ (không cần chạy lại thí nghiệm).

python -m analysis.report_figures                 # ghi PNG vào analysis/figures/
python -m analysis.report_figures --only heatmap  # chỉ vẽ các hình có tên chứa "heatmap"

Mỗi hình đọc đúng file log đã commit, nên ai chạy lại cũng ra cùng hình (Elo bootstrap cố định seed 0).
Hình nào thiếu file log thì bỏ qua và in cảnh báo.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analysis.arena_stats import bootstrap_elo, color_table, pair_table, timing_table, wilson_interval  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "analysis" / "figures"
DPI = 200
N_BOOT = 300

# Tên hiển thị thống nhất trong báo cáo
DISPLAY = {
    "random": "Cấp 1\nRandom", "heuristic": "Cấp 2\nHeuristic", "alphabeta": "Cấp 3\nAlpha-Beta",
    "mcts": "Cấp 4\nMCTS", "sl": "Cấp 5\nSL", "rl": "Cấp 6\nRL", "puct": "Cấp 7\nPUCT",
    "alphazero": "Cấp 8\nAZ",
}
COLORS = {"heuristic": "#1f77b4", "sl": "#d62728", "mcts": "#2ca02c", "policy": "#d62728", "puct": "#9467bd"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # bỏ nhãn vạch phụ kiểu "6×10²" trên trục log
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print(f"  ✓ {name}.png")


def score_of(games: list[dict[str, Any]], a: str, b: str) -> tuple[float, float, float, int]:
    row = pair_table(games)[(a, b)]
    sc = row["score"]
    # Wilson có thể lệch vài 1e-17 quanh 0 hoặc 1; kẹp lại để thanh sai số không âm
    return sc, min(sc, row["ci95"][0]), max(sc, row["ci95"][1]), row["n"]


# ---------------------------------------------------------------- giải đấu chính thức

def _tournament(name: str, title: str) -> None:
    path = RES / "issue15" / f"{name}.jsonl.gz"
    games = load_jsonl(path)
    elo = bootstrap_elo(games, n_boot=N_BOOT)
    order = sorted(elo, key=lambda a: -elo[a]["elo"])
    table = pair_table(games)

    # Heatmap ma trận đối đầu
    n = len(order)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if (a, b) in table:
                mat[i, j] = table[(a, b)]["score"]
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if 0.2 < mat[i, j] < 0.8 else "white")
    labels = [DISPLAY.get(a, a) for a in order]
    ax.set_xticks(range(n), labels, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("Đối thủ (cột)")
    ax.set_ylabel("Tác tử (hàng)")
    ax.set_title(f"Ma trận score hàng gặp cột – {title}\n(200 ván/cặp × số seed, xếp theo Elo)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="score")
    save(fig, f"heatmap_{name}")

    # Elo kèm khoảng tin cậy
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [elo[a]["elo"] for a in order]
    err = np.array([[elo[a]["elo"] - elo[a]["ci95"][0], elo[a]["ci95"][1] - elo[a]["elo"]] for a in order]).T
    ax.bar(range(n), vals, yerr=err, capsize=4, color="#4c72b0")
    ax.set_xticks(range(n), labels, fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Elo (Bradley-Terry, trung bình = 0)")
    ax.set_title(f"Xếp hạng Elo – {title} (CI 95% bootstrap {N_BOOT} lần)")
    for i, v in enumerate(vals):
        ax.text(i, v + (25 if v >= 0 else -60), f"{v:.0f}", ha="center", fontsize=8)
    save(fig, f"elo_{name}")

    # Sức mạnh so với chi phí
    timing = timing_table(games)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    placed: list[tuple[float, float]] = []
    for a in sorted(order, key=lambda a: -elo[a]["elo"]):
        t = timing[a]["mean_ms"]
        ax.errorbar(t, elo[a]["elo"], yerr=[[elo[a]["elo"] - elo[a]["ci95"][0]], [elo[a]["ci95"][1] - elo[a]["elo"]]],
                    fmt="o", capsize=3, color="#4c72b0")
        # Đẩy nhãn xuống nếu trùng chỗ một nhãn đã đặt (cùng khoảng thời gian, Elo gần nhau)
        dy = 4
        for pt, pe in placed:
            if abs(np.log10(t) - np.log10(pt)) < 0.4 and abs(elo[a]["elo"] - pe) < 90:
                dy = -14
        placed.append((t, elo[a]["elo"]))
        ax.annotate(DISPLAY.get(a, a).replace("\n", " "), (t, elo[a]["elo"]), textcoords="offset points",
                    xytext=(6, dy), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Thời gian trung bình mỗi nước (ms, thang log)")
    ax.set_ylabel("Elo")
    ax.set_title(f"Sức mạnh và chi phí suy luận – {title}")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, f"elo_vs_time_{name}")

    # Lợi thế đi trước theo tác tử
    colors = color_table(games)
    per = colors["agents"]
    if per:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(n)
        sx = [per[a]["score_as_x"] for a in order]
        so = [per[a]["score_as_o"] for a in order]
        ax.bar(x - 0.2, sx, 0.4, label="cầm X (đi trước)")
        ax.bar(x + 0.2, so, 0.4, label="cầm O (đi sau)")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Score theo quyền đi trước – {title} (X chung: {colors['x_score']:.3f})")
        ax.legend()
        save(fig, f"first_move_{name}")


def fig_mode_a() -> None:
    _tournament("mode_a_0.5s", "Chế độ A (0,5 s/nước)")


def fig_mode_b() -> None:
    _tournament("mode_b_default", "Chế độ B (cấu hình mặc định)")


# ---------------------------------------------------------------- RQ1 và ablation issue #8

def fig_rq1() -> None:
    rows = list(csv.DictReader(open(RES / "issue8" / "pairs.csv", encoding="utf-8")))
    timing = list(csv.DictReader(open(RES / "issue8" / "timing.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    rq1 = [r for r in rows if r["experiment"].startswith("rq1_")]
    budgets = [float(r["experiment"].split("_")[1].rstrip("s")) for r in rq1]
    sc = [float(r["score_a"]) for r in rq1]
    lo = [float(r["wilson_low"]) for r in rq1]
    hi = [float(r["wilson_high"]) for r in rq1]
    ax.errorbar(budgets, sc, yerr=[np.subtract(sc, lo), np.subtract(hi, sc)], fmt="o-", capsize=4, label="Alpha-Beta gặp MCTS")
    ax.set_xscale("log")
    ax.set_xticks(budgets, [f"{b:g} s" for b in budgets])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Ngân sách thời gian mỗi nước")
    ax.set_ylabel("score của Alpha-Beta")
    ax.set_title("RQ1: cùng thời gian suy nghĩ (600 ván/mức)")
    ax.legend(loc="lower right")

    ax = axes[1]
    sims = [r for r in rows if r["experiment"] == "ablation_simulations"]
    ns = [int(r["agent_b"].split("_")[1]) for r in sims]
    mcts_score = [1 - float(r["score_a"]) for r in sims]
    lo = [1 - float(r["wilson_high"]) for r in sims]
    hi = [1 - float(r["wilson_low"]) for r in sims]
    ax.errorbar(ns, mcts_score, yerr=[np.subtract(mcts_score, lo), np.subtract(hi, mcts_score)], fmt="s-",
                capsize=4, color=COLORS["mcts"], label="MCTS gặp Alpha-Beta độ sâu 3")
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(v) for v in ns])
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("Số mô phỏng MCTS mỗi nước")
    ax.set_ylabel("score của MCTS")
    ax2 = ax.twinx()
    t_by = {r["agent"]: float(r["mean_ms"]) for r in timing if r["experiment"] == "ablation_simulations"}
    ax2.plot(ns, [t_by.get(f"mcts_{v}", np.nan) for v in ns], "d:", color="gray", label="ms/nước")
    ax2.set_yscale("log")
    ax2.set_ylabel("ms/nước (thang log)", color="gray")
    ax.set_title("Ablation số mô phỏng MCTS (600 ván/mức)")
    ax.legend(loc="upper left")
    save(fig, "rq1_ablation_mcts")


def fig_ab_pruning() -> None:
    rows = list(csv.DictReader(open(RES / "issue8" / "alphabeta.csv", encoding="utf-8")))
    agg: dict[tuple[int, bool, bool], list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        agg[(int(r["depth"]), r["pruning"] == "True", r["ordering"] == "True")].append((int(r["nodes"]), float(r["time_used"])))
    depths = sorted({k[0] for k in agg})
    variants = [(False, False, "không cắt tỉa"), (True, False, "cắt tỉa"), (True, True, "cắt tỉa + sắp nước")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    w = 0.27
    for k, (pr, od, lab) in enumerate(variants):
        nodes = [np.mean([v[0] for v in agg[(d, pr, od)]]) for d in depths]
        times = [np.mean([v[1] for v in agg[(d, pr, od)]]) * 1000 for d in depths]
        x = np.arange(len(depths)) + (k - 1) * w
        axes[0].bar(x, nodes, w, label=lab)
        axes[1].bar(x, times, w, label=lab)
        for xi, v in zip(x, nodes):
            axes[0].text(xi, v * 1.1, f"{v:,.0f}", ha="center", fontsize=7, rotation=90)
    for ax, yl in ((axes[0], "số nút duyệt TB / nước"), (axes[1], "thời gian TB / nước (ms)")):
        ax.set_yscale("log")
        ax.set_xticks(range(len(depths)), [f"độ sâu {d}" for d in depths])
        ax.set_ylabel(yl + " (thang log)")
        ax.legend(fontsize=8)
    axes[0].set_title("Alpha-Beta: số nút có/không cắt tỉa")
    axes[1].set_title("Alpha-Beta: thời gian có/không cắt tỉa")
    fig.suptitle("Cùng nước đi và cùng giá trị ở mọi biến thể (kiểm tra trên 10 thế cờ)", fontsize=9, y=0.02)
    save(fig, "ab_pruning")


# ---------------------------------------------------------------- RQ2

def fig_rq2() -> None:
    games = load_jsonl(RES / "issue15" / "rq2_puct.jsonl.gz")
    timing = timing_table(games)
    sims = [400, 800, 1600]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    w = 0.27
    for k, (a, b, lab, c) in enumerate((("puct", "mcts", "PUCT gặp MCTS", COLORS["puct"]),
                                        ("puct", "policy", "PUCT gặp policy", "#8c564b"),
                                        ("policy", "mcts", "policy gặp MCTS", COLORS["policy"]))):
        vals = [score_of(games, f"{a}_{s}" if a != "policy" else "policy", f"{b}_{s}" if b != "policy" else "policy")
                for s in sims]
        x = np.arange(len(sims)) + (k - 1) * w
        ax.bar(x, [v[0] for v in vals], w, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
               capsize=3, label=lab, color=c)
        for xi, v in zip(x, vals):
            ax.text(xi, 0.5, f"{v[0]:.3f}  (n={v[3]})", ha="center", va="center", rotation=90, fontsize=7.5, color="white")
    ax.set_xticks(range(len(sims)), [f"{s} mô phỏng" for s in sims])
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("score của bên trước (Wilson 95%)")
    ax.set_title("RQ2: thứ tự thực nghiệm PUCT > policy > MCTS")
    ax.legend(loc="upper center", ncol=3, fontsize=7.5)
    ax = axes[1]
    ax.plot(sims, [timing[f"puct_{s}"]["mean_ms"] for s in sims], "o-", color=COLORS["puct"], label="PUCT (cấp 7)")
    ax.plot(sims, [timing[f"mcts_{s}"]["mean_ms"] for s in sims], "s-", color=COLORS["mcts"], label="MCTS (cấp 4)")
    ax.axhline(timing["policy"]["mean_ms"], color=COLORS["policy"], ls="--", label="policy (1 lần suy luận)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sims, [str(s) for s in sims])
    ax.set_xlabel("Số mô phỏng mỗi nước")
    ax.set_ylabel("ms/nước (thang log)")
    ax.set_title("Chi phí suy luận theo số mô phỏng")
    ax.legend()
    save(fig, "rq2_puct")


# ---------------------------------------------------------------- độ sâu Alpha-Beta

def fig_ab_depth() -> None:
    games = load_jsonl(RES / "issue15" / "ab_depth.jsonl.gz")
    timing = timing_table(games)
    depths = [2, 3, 4]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for opp, lab in (("heuristic", "gặp Heuristic (cấp 2)"), ("sl", "gặp SL (cấp 5)"), ("puct", "gặp PUCT 400 (cấp 7)")):
        vals = [score_of(games, f"ab_d{d}", opp) for d in depths]
        ax.errorbar(depths, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="o-", capsize=4, label=lab, color=COLORS.get(opp))
    ax.set_xticks(depths, [f"độ sâu {d}" for d in depths])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("score của Alpha-Beta (Wilson 95%)")
    ax.set_title("Ảnh hưởng độ sâu tìm kiếm (200 ván/cặp)")
    ax.legend()
    ax = axes[1]
    nodes = [timing[f"ab_d{d}"]["mean_nodes"] for d in depths]
    times = [timing[f"ab_d{d}"]["mean_ms"] for d in depths]
    p95 = [timing[f"ab_d{d}"]["p95_ms"] for d in depths]
    ax.bar(np.array(depths) - 0.2, nodes, 0.4, color="#4c72b0", label="số nút TB/nước")
    ax.set_yscale("log")
    ax.set_ylabel("số nút (thang log)")
    ax2 = ax.twinx()
    ax2.plot(depths, times, "o-", color="#dd8452", label="ms TB/nước")
    ax2.plot(depths, p95, "o--", color="#dd8452", alpha=0.6, label="ms p95")
    ax2.set_yscale("log")
    ax2.set_ylabel("ms/nước (thang log)")
    ax.set_xticks(depths, [f"độ sâu {d}" for d in depths])
    ax.set_title("Chi phí theo độ sâu")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    save(fig, "ab_depth")


# ---------------------------------------------------------------- learning curves

def fig_level5() -> None:
    recs = load_jsonl(RES / "training" / "level5" / "train_log.jsonl")
    summ = json.loads((RES / "training" / "level5" / "summary.json").read_text(encoding="utf-8"))
    ep = [r["epoch"] for r in recs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for key, ax, title in (("policy_loss", axes[0], "Policy loss"), ("value_loss", axes[1], "Value loss")):
        ax.plot(ep, [r["train"][key] for r in recs], label="train")
        ax.plot(ep, [r["val"][key] for r in recs], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    ax = axes[2]
    for key, lab in (("top1_acc", "top-1"), ("top3_acc", "top-3")):
        ax.plot(ep, [r["val"][key] for r in recs], label=f"val {lab}")
    ax.axhline(summ["test_metrics"]["top1_acc"], color="C0", ls=":", label=f"test top-1 {summ['test_metrics']['top1_acc']:.3f}")
    ax.axhline(summ["test_metrics"]["top3_acc"], color="C1", ls=":", label=f"test top-3 {summ['test_metrics']['top3_acc']:.3f}")
    ax.set_title("Độ chính xác nước đi")
    ax.set_xlabel("epoch")
    for ax in axes:
        ax.axvline(summ["best_epoch"], color="gray", ls="--", lw=0.8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Cấp 5 – học có giám sát: tốt nhất epoch {summ['best_epoch']}/{summ['epochs_ran']}, "
                 f"{summ['total_training_time_seconds']:.0f} s", fontsize=10)
    save(fig, "learning_level5")


def fig_level6() -> None:
    recs = load_jsonl(RES / "training" / "level6" / "train_log.jsonl")
    rnd = [r["round"] for r in recs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    sc = [r["score_vs_sl"] for r in recs]
    lo = [r["ci95_vs_sl"][0] for r in recs]
    hi = [r["ci95_vs_sl"][1] for r in recs]
    ax.plot(rnd, sc, "o-", label="score gặp SL (100 ván/vòng)")
    ax.fill_between(rnd, lo, hi, alpha=0.2, label="Wilson 95%")
    best = [r["round"] for r in recs if r.get("is_best")][-1]
    ax.axvline(best, color="gray", ls="--", lw=0.8, label=f"best: vòng {best}")
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xlabel("vòng self-play (1.000 episode/vòng)")
    ax.set_ylabel("score")
    ax.set_title("Cấp 6 – score khi gặp mạng SL ban đầu")
    ax.legend(fontsize=8)
    ax = axes[1]
    upd = [r for r in recs if r.get("entropy") is not None]
    x = [r["round"] for r in upd]
    ax.plot(x, [r["value_loss"] for r in upd], "s-", label="value loss")
    ax.plot(x, [r["entropy"] for r in upd], "^-", label="entropy")
    ax.plot(x, [r["kl"] for r in upd], "d-", label="KL(π‖π_SL)")
    ax.set_xlabel("vòng self-play")
    ax.set_title("Cấp 6 – động lực học Actor-Critic")
    ax.legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    save(fig, "learning_level6")


def fig_level8() -> None:
    recs = load_jsonl(RES / "training" / "level8" / "train_log.jsonl")
    rnd = [r["round"] for r in recs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(rnd, [r["policy_loss"] for r in recs], "o-", label="policy loss")
    ax.plot(rnd, [r["value_loss"] for r in recs], "s-", label="value loss")
    ax.set_xlabel("vòng (200 ván self-play/vòng)")
    ax.set_title("Cấp 8 – loss huấn luyện")
    ax.legend()
    ax = axes[1]
    sc = [r["gate_score"] for r in recs]
    lo = [r["gate_ci95"][0] for r in recs]
    hi = [r["gate_ci95"][1] for r in recs]
    ax.plot(rnd, sc, "o-", label="score gating gặp bản tốt nhất trước đó (40 ván)")
    ax.fill_between(rnd, lo, hi, alpha=0.2, label="Wilson 95%")
    ax.axhline(0.55, color="red", ls="--", lw=0.8, label="ngưỡng chấp nhận 0,55")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("vòng")
    ax.set_title(f"Cấp 8 – gating: chấp nhận {sum(r['accepted'] for r in recs)}/{len(recs)} vòng")
    ax.legend(fontsize=7, loc="lower right")
    for a in axes:
        a.grid(alpha=0.3)
    save(fig, "learning_level8")


def fig_rq3() -> None:
    games = load_jsonl(RES / "issue11" / "rq3_rl_vs_sl.jsonl.gz")
    rounds = [5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(7, 4))
    for opp, lab, c in (("sl", "RL gặp SL (cấp 5)", COLORS["sl"]), ("heuristic", "RL gặp Heuristic (cấp 2)", COLORS["heuristic"])):
        vals = [score_of(games, f"rl_r{r:03d}", opp) for r in rounds]
        ax.errorbar(rounds, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="o-", capsize=4, label=lab, color=c)
    base = score_of(games, "sl", "heuristic")
    ax.axhline(base[0], color=COLORS["heuristic"], ls=":", label=f"SL gặp Heuristic = {base[0]:.3f}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(rounds)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("vòng self-play của checkpoint cấp 6")
    ax.set_ylabel("score (Wilson 95%, 200 ván/cặp)")
    ax.set_title("RQ3: self-play có cải thiện so với mạng học có giám sát?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "rq3_rl_vs_sl")


def fig_az_ladder() -> None:
    games = load_jsonl(RES / "issue14" / "az_ladder.jsonl.gz")
    rounds = [1, 5, 10, 15, 20, 25]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for prefix, opp, lab, style in (("az_s", "heuristic", "PUCT 100 gặp Heuristic", "o-"),
                                    ("az_s", "sl", "PUCT 100 gặp SL", "s-"),
                                    ("az_p", "heuristic", "policy-only gặp Heuristic", "o--"),
                                    ("az_p", "sl", "policy-only gặp SL", "s--")):
        vals = [score_of(games, f"{prefix}{r:02d}", opp) for r in rounds]
        ax.errorbar(rounds, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt=style, capsize=3, label=lab, color=COLORS[opp])
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(rounds)
    ax.set_ylim(-0.01, 0.5)
    ax.set_xlabel("vòng huấn luyện cấp 8")
    ax.set_ylabel("score (Wilson 95%, 200 ván/cặp)")
    ax.set_title("Cấp 8 – đối đầu checkpoint với đối thủ cố định")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "az_ladder")


FIGURES: dict[str, Callable[[], None]] = {
    "mode_a": fig_mode_a, "mode_b": fig_mode_b, "rq1": fig_rq1, "ab_pruning": fig_ab_pruning,
    "rq2": fig_rq2, "ab_depth": fig_ab_depth, "level5": fig_level5, "level6": fig_level6,
    "level8": fig_level8, "rq3": fig_rq3, "az_ladder": fig_az_ladder,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh hình Chương 4 từ results/.")
    parser.add_argument("--only", nargs="*", default=None, help="Chỉ vẽ nhóm hình có tên này")
    args = parser.parse_args()
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}]")
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  ! bỏ qua: thiếu {exc.filename}")


if __name__ == "__main__":
    main()
