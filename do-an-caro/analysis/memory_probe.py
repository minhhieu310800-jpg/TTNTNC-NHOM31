"""Đo bộ nhớ đỉnh (peak RSS) của từng tác tử trong một file cấu hình giải đấu.

Mỗi tác tử chạy trong một tiến trình con riêng để số đo không bị lẫn giữa các tác tử:
  1. nạp thư viện và engine, ghi RSS đỉnh nền (baseline);
  2. dựng tác tử, cho gặp đối thủ nhẹ (mặc định Heuristic, gần như không tốn RAM) trên N khai cuộc
     đầu tiên, mỗi khai cuộc 2 ván đổi bên, để ván kéo dài tới trung cuộc như khi thi đấu thật;
  3. ghi RSS đỉnh sau khi chơi. Phần tăng thêm = đỉnh - nền là bộ nhớ do tác tử dùng
     (mạng nơ-ron, cây tìm kiếm, bảng chuyển vị...).

RSS đỉnh lấy từ resource.getrusage trên macOS/Linux và PeakWorkingSetSize (GetProcessMemoryInfo) trên Windows.
Chạy trên CPU nên không đo VRAM.

python -m analysis.memory_probe configs/arena/mode_b.yaml --openings 3 --out results/memory/mode_b.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
from pathlib import Path
from typing import Any

MB = 1024 * 1024


def _peak_rss_mb() -> float:
    """RSS đỉnh của tiến trình hiện tại (MB) trên macOS, Linux và Windows, không cần thư viện ngoài."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo thất bại")
        return counters.PeakWorkingSetSize / MB

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / MB if sys.platform == "darwin" else peak / 1024


def _probe(label: str, agent_cfg: dict[str, Any], cfg: dict[str, Any], n_openings: int, opponent: str,
           queue: mp.Queue) -> None:
    import time

    import torch  # noqa: F401  nạp sẵn để baseline đã gồm PyTorch, công bằng giữa các tác tử

    from arena.agent_factory import AgentSpec
    from arena.match import GameTask, play_game
    from arena.openings import load_openings
    from game.constants import DEFAULT_BOARD_SIZE, RULE_VIETNAM

    torch.set_num_threads(1)
    openings = load_openings(cfg["openings_file"])[:n_openings]
    size = int(cfg.get("size", DEFAULT_BOARD_SIZE))
    rule = str(cfg.get("rule", RULE_VIETNAM))
    spec = AgentSpec.from_config(label, agent_cfg)
    rand = AgentSpec.from_config(opponent, {"type": opponent})
    baseline = _peak_rss_mb()

    moves = 0
    t0 = time.perf_counter()
    for op in openings:
        for swap in (0, 1):
            x, o = (spec, rand) if swap == 0 else (rand, spec)
            rec = play_game(GameTask(
                game_id=f"mem_{label}_{op['id']}_{swap}", experiment="memory_probe", mode=cfg["mode"],
                budget=cfg.get("budget"), size=size, rule=rule, opening_id=int(op["id"]),
                opening_moves=tuple(int(m) for m in op["moves"]), seed=0,
                agent_x=x, agent_o=o, seed_x=0, seed_o=1,
            ))
            if rec.get("error"):
                raise RuntimeError(f"{label}: {rec['error']}")
            side = 1 if swap == 0 else -1
            moves += sum(1 for m in rec["moves"] if m["player"] == side)
    peak = _peak_rss_mb()
    queue.put({
        "agent": label,
        "type": agent_cfg["type"],
        "params": agent_cfg.get("params", {}),
        "games": 2 * len(openings),
        "opponent": opponent,
        "agent_moves": moves,
        "baseline_rss_mb": round(baseline, 1),
        "peak_rss_mb": round(peak, 1),
        "agent_extra_mb": round(peak - baseline, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
    })


def probe_config(cfg: dict[str, Any], n_openings: int, only: list[str] | None = None,
                 opponent: str = "heuristic") -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    rows = []
    for label, agent_cfg in cfg["agents"].items():
        if only and label not in only:
            continue
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_probe, args=(label, agent_cfg, cfg, n_openings, opponent, queue))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Tiến trình đo {label} lỗi (exit {proc.exitcode}).")
        row = queue.get()
        print(f"{row['agent']:>10}: nền {row['baseline_rss_mb']:7.1f} MB | đỉnh {row['peak_rss_mb']:7.1f} MB "
              f"| tăng {row['agent_extra_mb']:6.1f} MB | {row['agent_moves']} nước, {row['wall_s']} s")
        rows.append(row)
    return rows


def _train_step_probe(blocks: int, channels: int, batch: int, queue: mp.Queue) -> None:
    """RSS đỉnh của một bước huấn luyện (forward + backward + Adam) trên CPU, thay cho VRAM không ghi được trên Colab."""
    import torch
    import torch.nn.functional as F

    from agents.network import PolicyValueNet

    torch.set_num_threads(1)
    torch.manual_seed(0)
    baseline = _peak_rss_mb()
    net = PolicyValueNet(blocks=blocks, channels=channels, size=11)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    x = (torch.rand(batch, 4, 11, 11) < 0.2).float()
    target_p = torch.randint(0, 121, (batch,))
    target_v = torch.rand(batch) * 2 - 1
    for _ in range(3):
        logits, value = net(x)
        loss = F.cross_entropy(logits, target_p) + F.mse_loss(value.view(-1), target_v)
        opt.zero_grad()
        loss.backward()
        opt.step()
    queue.put({
        "blocks": blocks, "channels": channels, "batch": batch,
        "params": int(sum(p.numel() for p in net.parameters())),
        "baseline_rss_mb": round(baseline, 1), "peak_rss_mb": round(_peak_rss_mb(), 1),
        "train_extra_mb": round(_peak_rss_mb() - baseline, 1),
    })


def probe_training(nets: list[tuple[str, int, int]], batch: int = 256) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    rows = []
    for name, blocks, channels in nets:
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_train_step_probe, args=(blocks, channels, batch, queue))
        proc.start()
        proc.join()
        row = {"network": name, **queue.get()}
        print(f"{name:>8}: {row['params']:,} tham số | tăng {row['train_extra_mb']} MB khi train batch {batch}")
        rows.append(row)
    return rows


def to_markdown(rows: list[dict[str, Any]], experiment: str) -> str:
    lines = [
        f"# Bộ nhớ đỉnh theo tác tử ({experiment})",
        "",
        f"Máy: {platform.platform()}, Python {platform.python_version()}. Mỗi tác tử chạy trong tiến trình riêng, "
        "1 luồng PyTorch; RSS nền đã gồm Python, NumPy, PyTorch và engine.",
        "",
        "| tác tử | loại | số ván | số nước | RSS nền (MB) | RSS đỉnh (MB) | tăng do tác tử (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['agent']} | {r['type']} | {r['games']} | {r['agent_moves']} | {r['baseline_rss_mb']} "
                     f"| {r['peak_rss_mb']} | {r['agent_extra_mb']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from arena.tournament import load_config

    parser = argparse.ArgumentParser(description="Đo RAM đỉnh của từng tác tử trong file cấu hình giải đấu.")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--train", action="store_true", help="Đo RAM một bước huấn luyện của mạng cấp 5/6 và cấp 8")
    parser.add_argument("--openings", type=int, default=3, help="Số khai cuộc (mỗi khai cuộc 2 ván đổi bên)")
    parser.add_argument("--opponent", default="heuristic", choices=["heuristic", "random"])
    parser.add_argument("--agents", nargs="*", default=None, help="Chỉ đo các tác tử này")
    parser.add_argument("--out", default=None, help="File JSON kết quả (kèm file .md cùng tên)")
    args = parser.parse_args()

    if args.train:
        rows = probe_training([("cap5_6", 6, 64), ("cap8", 4, 64)])
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"platform": platform.platform(), "device": "cpu", "rows": rows}, indent=2,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
        return
    cfg = load_config(args.config)
    rows = probe_config(cfg, args.openings, args.agents, args.opponent)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": args.config, "openings": args.openings, "platform": platform.platform(),
                "python": platform.python_version(), "opponent": args.opponent, "rows": rows}
        out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out.with_suffix(".md").write_text(to_markdown(rows, cfg["name"]), encoding="utf-8")
        print(f"Đã ghi {out} và {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
