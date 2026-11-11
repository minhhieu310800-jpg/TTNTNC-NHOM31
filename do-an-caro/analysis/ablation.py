"""Thí nghiệm tái lập cho issue #8, chạy từ thư mục do-an-caro.

Có hỗ trợ từ OpenAI Codex; dùng lại arena và thống kê của dự án.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
from pathlib import Path

import yaml

from agents.alphabeta_agent import AlphaBetaAgent
from agents.fast_eval import fast_evaluate
from arena.openings import load_openings
from arena.runtime import keep_awake
from arena.tournament import build_schedule, run_tournament
from analysis.arena_stats import load_games
from analysis.summarize import pairs_markdown, timing_markdown
from game.game import CaroGame

SIMULATIONS = (100, 400, 1600, 6400)
BUDGETS = (0.1, 0.5, 1.0)


def experiment_configs(smoke: bool = False) -> list[dict]:
    """Dùng chung khai cuộc và seed, đổi bên cho mỗi cặp so sánh."""
    common = dict(size=11, rule="vietnam", openings_file="configs/openings.json",
                  num_openings=1 if smoke else 100, seeds=[0] if smoke else [0, 1, 2],
                  workers=1)
    suffix = "_smoke" if smoke else ""
    baseline = {"type": "alphabeta", "params": {"depth": 3}}
    sims_agents = {"ab": baseline}
    for n in SIMULATIONS:
        sims_agents[f"mcts_{n}"] = {"type": "mcts", "params": {
            "simulations": n, "rollout": "random", "use_forced_move": False}}
    configs = [dict(common, name="ablation_simulations" + suffix, mode="B", budget=None,
                    agents=sims_agents, pairs=[["ab", f"mcts_{n}"] for n in SIMULATIONS])]
    configs.append(dict(common, name="ablation_rollout" + suffix, mode="B", budget=None,
                        agents={f"mcts_{r}": {"type": "mcts", "params": {
                            "simulations": 20 if smoke else 1600,
                            "rollout": r, "use_forced_move": False}}
                            for r in ("random", "heuristic")}))
    for budget in BUDGETS:
        configs.append(dict(common, name=f"rq1_{budget:.1f}s" + suffix, mode="A", budget=budget,
                            agents={"ab": baseline, "mcts": {"type": "mcts", "params": {
                                "rollout": "random", "use_forced_move": False}}}))
    return configs


def benchmark_ab(openings: list[dict], depth: int) -> list[dict]:
    """Cố định độ sâu, tắt TT, không giới hạn giờ khi đo cắt tỉa và sắp nước."""
    fast_evaluate(CaroGame().board, 1)  # Biên dịch Numba trước khi đo.
    rows = []
    for opening in openings:
        game = CaroGame()
        for action in opening["moves"]:
            game.step(action)
        group = []
        for pruning, ordering in itertools.product((False, True), repeat=2):
            agent = AlphaBetaAgent(depth=depth, use_pruning=pruning,
                                  use_move_ordering=ordering, use_tt=False)
            action = agent.select_action(game.clone())
            group.append(dict(opening_id=opening["id"], depth=depth, pruning=pruning,
                              ordering=ordering, action=action, **agent.last_stats))
        reference = group[0]
        for row in group:
            row["same_action"] = row["action"] == reference["action"]
            row["same_score"] = row["score"] == reference["score"]
            rows.append(row)
    return rows


def write_report(out: Path, rows: list[dict], configs: list[dict]) -> None:
    lines = ["# Ablation Alpha-Beta/MCTS và RQ1", "",
             "Alpha-Beta: cùng vị trí, độ sâu cố định, radius=2, fast_eval=True, TT=False. "
             "Mốc so sánh: tắt cả cắt tỉa và sắp nước. Khác nước nhưng cùng điểm có thể do hòa điểm.",
             "", "| Vị trí | Sâu | Cắt tỉa | Sắp nước | Nước | Điểm | Nút | Giây | Cùng nước | Cùng điểm |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['opening_id']} | {r['depth']} | {r['pruning']} | {r['ordering']} | "
                     f"{r['action']} | {r['score']} | {r['nodes']} | {r['time_used']:.6f} | "
                     f"{r['same_action']} | {r['same_score']} |")
    if any(not r["same_score"] for r in rows):
        lines += ["", "CẢNH BÁO: có khác biệt điểm minimax; cần kiểm tra trước khi kết luận."]
    lines += ["", "MCTS thuần không bật forced move. Số mô phỏng chỉ cố định trong mode B. "
              "RQ1 dùng mode A, ngân sách thời gian thay cho số mô phỏng. "
              "Đây là ngân sách mềm: ghi thời gian thực tế để kiểm tra vượt ngân sách. "
              "Wilson 95% cho score có hòa là xấp xỉ; các ván dùng chung khai cuộc/seed có tương quan."]
    for cfg in configs:
        games = load_games(str(out / f"{cfg['name']}.jsonl"))
        expected = len(build_schedule(cfg, load_openings(cfg["openings_file"])))
        errors = sum(g.get("error") is not None for g in games)
        lines += ["", f"## {cfg['name']}",
                  f"Hoàn thành {len(games)}/{expected} ván; lỗi tác tử: {errors}."]
        if len(games) != expected or errors or cfg["name"].endswith("_smoke"):
            lines += ["Chỉ dùng kiểm tra pipeline/chạy chưa đủ hoặc có lỗi; chưa dùng kết luận RQ1."]
        if games:
            lines += ["", pairs_markdown(games), "", timing_markdown(games)]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="logs/ablation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ab-depth", type=int, nargs="+", default=[2, 3],
                        help="Các độ sâu Alpha-Beta cần đo (mặc định: 2 3)")
    parser.add_argument("--ab-positions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Số ván mỗi thí nghiệm trong một lần chạy; hỗ trợ chạy tiếp")
    parser.add_argument("--only", choices=("all", "ab", "simulations", "rollout", "rq1"), default="all")
    args = parser.parse_args()
    if any(d < 1 for d in args.ab_depth) or args.ab_positions < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Độ sâu, số vị trí, số worker và giới hạn ván phải dương")
    if len(set(args.ab_depth)) != len(args.ab_depth):
        parser.error("Các độ sâu không được trùng nhau")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = experiment_configs(args.smoke)
    if args.only not in ("all", "ab"):
        configs = [c for c in configs if args.only in c["name"]]
    if args.only == "ab":
        configs = []
    rows = []
    if args.only in ("all", "ab"):
        openings = load_openings("configs/openings.json")
        for depth in args.ab_depth:
            rows.extend(benchmark_ab(openings[:1 if args.smoke else args.ab_positions], depth))
        with (out / "alphabeta.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "arguments": vars(args)}, indent=2), encoding="utf-8")
    fast_evaluate(CaroGame().board, 1)
    for cfg in configs:
        path = out / f"{cfg['name']}.yaml"
        # Chỉ chạy tiếp cùng cấu hình vì mã ván không chứa tham số tác tử.
        if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != cfg:
            raise ValueError(f"Cấu hình đã đổi: chọn thư mục --out khác ({path})")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        run_tournament(cfg, log_dir=out, workers=args.workers, limit=args.limit)
    write_report(out, rows, configs)
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    with keep_awake():
        main()
