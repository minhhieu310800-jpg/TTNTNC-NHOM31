"""Phân tích lỗi chiến thuật và minh họa ván điển hình (đề bài mục 3.6, Chương 4).

1) Lỗi chiến thuật, đếm tự động trên MỌI nước đi của tác tử (không chọn ván):
   - bỏ lỡ thắng ngay : có ít nhất một nước thắng ngay nhưng tác tử đi nước khác;
   - không chặn       : không có nước thắng, đối thủ có ĐÚNG MỘT nước thắng ở lượt sau, tác tử không đi vào ô đó;
   - thế thua chắc    : đối thủ có từ hai nước thắng trở lên (không thể chặn hết), đếm riêng, không tính là lỗi.
   Một nước thắng phải nằm cạnh (8 hướng) ít nhất một quân của bên đi, nên chỉ thử các ô đó cho nhanh.

2) Minh họa ván: với mỗi cặp đã định sẵn, chọn theo luật cố định ván có độ dài TRUNG VỊ trong số ván
   tác tử thắng và trong số ván tác tử thua (không tự chọn ván đẹp), vẽ bàn cờ có số thứ tự nước
   và đánh dấu lỗi chiến thuật đầu tiên nếu có.

python -m analysis.game_analysis errors results/issue15/mode_b_default.jsonl.gz --seed 0 --out results/analysis/tactical_errors_mode_b.md
python -m analysis.game_analysis boards
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from game.constants import DEFAULT_BOARD_SIZE, RULE_VIETNAM
from game.game import CaroGame

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "analysis" / "figures"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- lỗi chiến thuật

def winning_moves(game: CaroGame, player: int) -> set[int]:
    """Các ô mà nếu `player` đặt quân vào thì thắng ngay (không đổi lượt thật của ván)."""
    size = game.size
    board = game.board
    own = np.argwhere(board == player)
    if own.size == 0:
        return set()
    cand: set[int] = set()
    for r, c in own:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0:
                    cand.add(int(rr * size + cc))
    wins = set()
    saved_player = game.current_player
    for a in cand:
        game.current_player = player
        game.step(a)
        if game.winner == player:
            wins.add(a)
        game.undo()
    game.current_player = saved_player
    return wins


def _candidates(game: CaroGame) -> set[int]:
    """Ô trống trong bán kính 2 quanh mọi quân (nơi có thể tạo đe dọa mới)."""
    size = game.size
    occ = np.argwhere(game.board != 0)
    cand: set[int] = set()
    for r, c in occ:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and game.board[rr, cc] == 0:
                    cand.add(int(rr * size + cc))
    return cand


def block_is_hopeless(game: CaroGame, p: int, block: int) -> bool:
    """Sau khi bên p chặn ở `block`, đối thủ có nước tạo >= 2 đường thắng ngay (bên p không thắng trước được)?

    Nếu đúng thì chặn cũng thua trong 3 nước, nên không chặn không phải lỗi quyết định.
    """
    saved = game.current_player
    game.current_player = p
    game.step(block)
    hopeless = False
    for m in _candidates(game):
        game.current_player = -p
        game.step(m)
        if game.winner is None and not winning_moves(game, p) and len(winning_moves(game, -p)) >= 2:
            hopeless = True
        game.undo()
        if hopeless:
            break
    game.undo()
    game.current_player = saved
    return hopeless


def scan_game(g: dict[str, Any]) -> dict[str, Any]:
    """Duyệt một ván, trả về số nước và lỗi theo từng bên."""
    game = CaroGame(size=int(g.get("size", DEFAULT_BOARD_SIZE)), rule=str(g.get("rule", RULE_VIETNAM)))
    for a in g["opening_moves"]:
        game.step(int(a))
    labels = {1: g["agent_x"]["label"], -1: g["agent_o"]["label"]}
    out = {lab: {"moves": 0, "missed_win": 0, "missed_block": 0, "hopeless_block": 0, "lost_position": 0,
                 "first_error": None}
           for lab in labels.values()}
    for i, mv in enumerate(g["moves"]):
        p = mv["player"]
        lab = labels[p]
        a = int(mv["action"])
        row = out[lab]
        row["moves"] += 1
        my_wins = winning_moves(game, p)
        err = None
        if my_wins:
            if a not in my_wins:
                row["missed_win"] += 1
                err = "bỏ lỡ thắng"
        else:
            threats = winning_moves(game, -p)
            if threats:
                # Chặn được bằng cách đi vào ô thắng HOẶC (luật Việt Nam) bịt đầu còn lại cho đường 5 bị chặn 2 đầu;
                # nên chỉ coi là không chặn khi sau nước đi đối thủ vẫn còn nước thắng ngay.
                game.step(a)
                still = bool(winning_moves(game, -p))
                game.undo()
                if still and len(threats) >= 2:
                    row["lost_position"] += 1
                elif still:
                    if block_is_hopeless(game, p, next(iter(threats))):
                        row["hopeless_block"] += 1      # chặn cũng thua trong 3 nước: không tính là lỗi
                    else:
                        row["missed_block"] += 1
                        err = "không chặn"
        if err and row["first_error"] is None:
            row["first_error"] = {"move_index": i, "kind": err, "action": a,
                                  "should": sorted(my_wins or winning_moves(game, -p))}
        game.step(a)
    return {"game_id": g["game_id"], "per_agent": out}


def tactical_errors(games: list[dict[str, Any]], workers: int = 4) -> tuple[dict[str, dict[str, int]], list[dict]]:
    with mp.get_context("spawn").Pool(workers) as pool:
        scans = pool.map(scan_game, games, chunksize=32)
    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in scans:
        for lab, row in s["per_agent"].items():
            for k in ("moves", "missed_win", "missed_block", "hopeless_block", "lost_position"):
                agg[lab][k] += row[k]
            agg[lab]["games"] += 1
            agg[lab]["games_with_error"] += int(row["missed_win"] + row["missed_block"] > 0)
    return {k: dict(v) for k, v in agg.items()}, scans


def errors_markdown(agg: dict[str, dict[str, int]], source: str, n_games: int) -> str:
    lines = [
        f"# Lỗi chiến thuật theo tác tử ({source}, {n_games} ván)",
        "",
        "Đếm trên mọi nước đi của tác tử. Bỏ lỡ thắng: có nước thắng ngay nhưng không đi. Không chặn: đối thủ có "
        "đúng một nước thắng ở lượt sau và sau nước đi của tác tử vẫn còn nước thắng (chặn hợp lệ gồm cả bịt đầu còn "
        "lại theo luật chặn 2 đầu), trong khi chặn thì chưa thua ngay. Chặn vô ích: không chặn nhưng dù chặn thì đối "
        "thủ vẫn tạo được hai đường thắng ở nước kế tiếp. Thế thua chắc: đối thủ đã có từ hai nước thắng. Hai loại "
        "sau không tính là lỗi.",
        "",
        "| tác tử | số ván | số nước | bỏ lỡ thắng | không chặn | lỗi / 100 nước | % ván có lỗi | chặn vô ích | thế thua chắc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for lab, r in sorted(agg.items(), key=lambda kv: (kv[1]["missed_win"] + kv[1]["missed_block"]) / max(kv[1]["moves"], 1)):
        err = r["missed_win"] + r["missed_block"]
        lines.append(
            f"| {lab} | {r['games']} | {r['moves']} | {r['missed_win']} | {r['missed_block']} | "
            f"{100 * err / max(r['moves'], 1):.2f} | {100 * r['games_with_error'] / max(r['games'], 1):.1f} | {r['hopeless_block']} | {r['lost_position']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- minh họa ván

# (file log, tác tử chính, đối thủ, tên hiển thị) – định sẵn trước khi xem ván
BOARD_CASES = [
    ("results/issue15/mode_b_default.jsonl.gz", "alphazero", "heuristic", "Cấp 8 gặp Heuristic (chế độ B)"),
    ("results/issue15/mode_b_default.jsonl.gz", "puct", "heuristic", "Cấp 7 PUCT gặp Heuristic (chế độ B)"),
    ("results/issue15/mode_b_default.jsonl.gz", "mcts", "sl", "Cấp 4 MCTS gặp Cấp 5 SL (chế độ B)"),
    ("results/issue15/ab_depth.jsonl.gz", "ab_d3", "puct", "Alpha-Beta độ sâu 3 gặp PUCT 400"),
    ("results/issue8/rq1_0.5s.jsonl.gz", "mcts", "ab", "RQ1: MCTS gặp Alpha-Beta (0,5 s/nước)"),
]


def median_game(games: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not games:
        return None
    games = sorted(games, key=lambda g: (len(g["moves"]), g["game_id"]))
    return games[(len(games) - 1) // 2]


def pick_cases(games: list[dict[str, Any]], me: str, opp: str) -> list[tuple[str, dict[str, Any]]]:
    pair = [g for g in games if {g["agent_x"]["label"], g["agent_o"]["label"]} == {me, opp} and g.get("seed", 0) == 0]
    won, lost = [], []
    for g in pair:
        side = 1 if g["agent_x"]["label"] == me else -1
        if g["result"] == side:
            won.append(g)
        elif g["result"] == -side:
            lost.append(g)
    out = []
    for tag, bucket in (("thắng", won), ("thua", lost)):
        g = median_game(bucket)
        if g is not None:
            out.append((f"{me} {tag} ({len(bucket)}/{len(pair)} ván {tag})", g))
    return out


def draw_board(ax, g: dict[str, Any], title: str) -> None:
    import matplotlib.patches as mpatches

    size = int(g.get("size", DEFAULT_BOARD_SIZE))
    scan = scan_game(g)
    err_moves = {}
    for lab, row in scan["per_agent"].items():
        if row["first_error"]:
            err_moves[row["first_error"]["move_index"]] = row["first_error"]
    ax.set_xlim(-0.6, size - 0.4)
    ax.set_ylim(size - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_facecolor("#f3d9a4")
    for i in range(size):
        ax.plot([0, size - 1], [i, i], color="#8b6b3d", lw=0.6)
        ax.plot([i, i], [0, size - 1], color="#8b6b3d", lw=0.6)
    ax.set_xticks(range(size), [chr(ord("A") + i) for i in range(size)], fontsize=7)
    ax.set_yticks(range(size), [str(i + 1) for i in range(size)], fontsize=7)
    player = 1
    for k, a in enumerate(g["opening_moves"]):
        r, c = divmod(int(a), size)
        ax.add_patch(mpatches.Circle((c, r), 0.42, color="black" if player == 1 else "white", ec="gray", lw=0.6))
        ax.text(c, r, "k", ha="center", va="center", fontsize=6, color="white" if player == 1 else "black")
        player = -player
    for i, mv in enumerate(g["moves"]):
        r, c = divmod(int(mv["action"]), size)
        p = mv["player"]
        ax.add_patch(mpatches.Circle((c, r), 0.42, color="black" if p == 1 else "white", ec="gray", lw=0.6))
        ax.text(c, r, str(i + 1), ha="center", va="center", fontsize=6.5 if i < 99 else 5,
                color="white" if p == 1 else "black")
        if i in err_moves:
            ax.add_patch(mpatches.Circle((c, r), 0.5, fill=False, ec="red", lw=1.8))
            for s in err_moves[i]["should"]:
                rr, cc = divmod(int(s), size)
                ax.add_patch(mpatches.Rectangle((cc - 0.45, rr - 0.45), 0.9, 0.9, fill=False, ec="red", lw=1.2, ls="--"))
    if g["moves"]:
        r, c = divmod(int(g["moves"][-1]["action"]), size)
        ax.add_patch(mpatches.Circle((c, r), 0.5, fill=False, ec="#1f77b4", lw=1.8))
    res = {1: "X thắng", -1: "O thắng", 0: "hòa"}[g["result"]]
    notes = [f"{v['kind']} ở nước {k + 1}" for k, v in sorted(err_moves.items())]
    sub = f"X: {g['agent_x']['label']} · O: {g['agent_o']['label']} · {res} · {len(g['moves'])} nước · khai cuộc #{g['opening_id']}"
    ax.set_title(f"{title}\n{sub}" + (f"\nlỗi: {', '.join(notes)}" if notes else ""), fontsize=8)


def make_boards() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict[str, Any]]] = {}
    written = []
    summary = []
    for idx, (src, me, opp, title) in enumerate(BOARD_CASES, start=1):
        if src not in cache:
            cache[src] = load_jsonl(ROOT / src)
        cases = pick_cases(cache[src], me, opp)
        if not cases:
            print(f"  ! không có ván cho {me} gặp {opp}")
            continue
        fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 6.2))
        axes = np.atleast_1d(axes)
        for ax, (tag, g) in zip(axes, cases):
            draw_board(ax, g, tag)
            summary.append({"case": title, "tag": tag, "game_id": g["game_id"], "source": src})
        fig.suptitle(f"{title} – ván có độ dài trung vị\n(k = quân khai cuộc; vòng đỏ = lỗi đầu tiên, ô đỏ nét đứt = "
                     f"nước đúng; vòng xanh = nước cuối)", fontsize=9)
        fig.tight_layout(rect=(0, 0.01, 1, 0.92))
        name = f"van_dien_hinh_{idx}_{me}_vs_{opp}.png"
        fig.savefig(FIG / name, dpi=200)
        plt.close(fig)
        written.append(name)
        print(f"  ✓ {name}")
    (ROOT / "results" / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "analysis" / "van_dien_hinh.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Lỗi chiến thuật và minh họa ván điển hình.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("errors")
    e.add_argument("log")
    e.add_argument("--seed", type=int, default=None, help="Chỉ lấy ván của seed này")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--out", required=True)
    sub.add_parser("boards")
    args = parser.parse_args()

    if args.cmd == "errors":
        games = load_jsonl(args.log)
        if args.seed is not None:
            games = [g for g in games if g.get("seed", 0) == args.seed]
        agg, _ = tactical_errors(games, args.workers)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(errors_markdown(agg, Path(args.log).name, len(games)), encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out.read_text(encoding="utf-8"))
    else:
        make_boards()


if __name__ == "__main__":
    main()
