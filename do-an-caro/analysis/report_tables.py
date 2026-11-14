"""Sinh bảng môi trường chạy và độ phức tạp trò chơi cho báo cáo (đề bài mục 3.1, 3.4, 3.5).

python -m analysis.report_tables   # ghi results/analysis/moi_truong_chay.md và do_phuc_tap.md

Thông tin phần cứng lấy từ file meta mà mỗi lần chạy tự ghi. Loại GPU khi huấn luyện không có trong
log; dòng GPU T4 lấy theo xác nhận của nhóm và được ghi chú rõ trong bảng.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "analysis"


def _jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cpu(h: dict) -> str:
    plat = h.get("platform", "")
    if plat.startswith("macOS"):
        return f"macOS, Apple M1 ({h.get('cpu_count')} nhân)"
    osname = "Windows 11" if "Windows-11" in plat else "Windows" if "Windows" in plat else plat.split("-")[0]
    return f"{osname}, Intel x86-64 ({h.get('cpu_count')} luồng)"


def _games(meta_runs: list[dict]) -> int:
    return max(int(m.get("games_total", 0)) for m in meta_runs)


def environment_rows() -> list[list[str]]:
    rows = []
    arena = [
        ("RQ1 + ablation MCTS/rollout (#8)", None),
        ("Giải chính thức chế độ A (0,5 s/nước)", RES / "issue15" / "mode_a_0.5s.meta.jsonl"),
        ("Giải chính thức chế độ B (mặc định)", RES / "issue15" / "mode_b_default.meta.jsonl"),
        ("RQ2: PUCT / policy / MCTS", RES / "issue15" / "rq2_puct.meta.jsonl"),
        ("Độ sâu Alpha-Beta 2/3/4", RES / "issue15" / "ab_depth.meta.jsonl"),
        ("RQ3: RL gặp SL và Heuristic (chạy lại)", RES / "issue11" / "rq3_rl_vs_sl.meta.jsonl"),
        ("Đối đầu checkpoint cấp 8", RES / "issue14" / "az_ladder.meta.jsonl"),
    ]
    for name, path in arena:
        if path is None:
            meta = json.loads((RES / "issue8" / "metadata.json").read_text(encoding="utf-8"))
            runs = [r for v in meta["run_metadata"].values() for r in v]
            h = runs[0]["hardware"]
            games = sum(_games([r]) for k, v in meta["run_metadata"].items() for r in v[:1])
            workers = runs[0]["workers"]
            py = h["python"]
        else:
            runs = _jsonl(path)
            h = runs[0]["hardware"]
            games = _games(runs)
            workers = runs[0]["workers"]
            py = h["python"]
        rows.append([name, _cpu(h), "CPU", str(workers), f"{games:,}".replace(",", "."), py])

    gen = json.loads((RES / "training" / "level5" / "dataset_sl_log.json").read_text(encoding="utf-8"))
    rows.append(["Sinh dữ liệu cấp 5 (20.000 ván Alpha-Beta)", _cpu(gen["hardware"]), "CPU",
                 str(gen["config"]["generation"]["workers"]), f"{gen['time_elapsed_seconds']:,.0f} s".replace(",", "."),
                 gen["hardware"]["python"]])
    l5 = json.loads((RES / "training" / "level5" / "summary.json").read_text(encoding="utf-8"))
    l6 = json.loads((RES / "training" / "level6" / "summary.json").read_text(encoding="utf-8"))
    l8 = _jsonl(RES / "training" / "level8" / "train_log.jsonl")
    l8_meta = _jsonl(RES / "training" / "level8" / "meta.jsonl")[0]
    t8 = sum(r["selfplay_s"] + r["train_s"] + r["gate_s"] for r in l8)
    gpu = "GPU NVIDIA T4 (*)"
    rows.append(["Huấn luyện cấp 5 (SL)", "Google Colab", gpu, "–", f"{l5['total_training_time_seconds']:,.0f} s".replace(",", "."), "–"])
    rows.append(["Huấn luyện cấp 6 (RL self-play)", "Google Colab", gpu, "–", f"{l6['total_time_seconds']:,.0f} s", "–"])
    rows.append([f"Huấn luyện cấp 8 ({len(l8)} vòng)", f"Google Colab ({l8_meta['hardware']['cpu_count']} CPU, "
                 f"{l8_meta['hardware']['ram_gb']} GB RAM)", gpu, "1", f"{t8:,.0f} s ({t8 / 3600:.2f} giờ)".replace(",", "."),
                 l8_meta["hardware"]["python"]])
    mem = json.loads((RES / "memory" / "mode_b.json").read_text(encoding="utf-8"))
    rows.append(["Đo RAM từng tác tử", "macOS, Apple M1 (8 nhân, 8 GB)", "CPU", "1", "–", mem["python"]])
    return rows


def environment_md() -> str:
    lines = [
        "# Môi trường chạy",
        "",
        "| công việc | máy | thiết bị tính | số worker | số ván / thời gian | Python |",
        "|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in environment_rows()]
    lines += [
        "",
        "(*) Log huấn luyện ghi `device: cuda` và thư mục `/content/drive/...` (Google Colab) nhưng không ghi tên GPU; "
        "loại GPU được xác nhận bằng phép đo VRAM trên cùng loại runtime Colab: Tesla T4, CUDA 13.0 "
        "(results/memory/train_vram_t4.json). Mọi tác tử dùng mạng nơ-ron khi thi đấu chạy trên CPU (`device: cpu`). "
        "Engine 1.0.0, NumPy 2.4.6 trên mọi máy chạy giải đấu.",
    ]
    return "\n".join(lines) + "\n"


def complexity_md() -> str:
    n = 121
    states = sum(math.comb(n, k) * math.comb(k, (k + 1) // 2) for k in range(n + 1))
    games = _jsonl(RES / "issue15" / "mode_b_default.jsonl.gz")
    no_random = [g for g in games if "random" not in (g["agent_x"]["label"], g["agent_o"]["label"])]
    plies = [len(g["opening_moves"]) + len(g["moves"]) for g in no_random]
    mean_len = st.mean(plies)
    d = round(mean_len)
    tree = sum(math.log10(n - k) for k in range(d))
    draws = sum(g["result"] == 0 for g in games)
    x_score = st.mean(1.0 if g["result"] == 1 else 0.5 if g["result"] == 0 else 0.0 for g in games)
    gen = json.loads((RES / "training" / "level5" / "dataset_sl_log.json").read_text(encoding="utf-8"))
    lines = [
        "# Đặc điểm và độ phức tạp của Caro 11×11 luật Việt Nam",
        "",
        "| đại lượng | giá trị | cách tính |",
        "|---|---|---|",
        "| số ô / số hành động tối đa | 121 | bàn 11×11, mỗi ô trống là một nước hợp lệ |",
        f"| cận trên số trạng thái | ≈ 10^{math.log10(states):.1f} | Σ_k C(121,k)·C(k,⌈k/2⌉): số cách đặt k quân với số X bằng số O hoặc hơn 1, chưa loại thế đã kết thúc |",
        f"| cận thô 3^121 | ≈ 10^{121 * math.log10(3):.1f} | mỗi ô trống/X/O |",
        f"| độ dài ván trung bình (không tính ván có Random) | {mean_len:.1f} nước (trung vị {st.median(plies):.0f}, tối đa {max(plies)}) | {len(no_random):,} ván chế độ B, gồm 4 nước khai cuộc |".replace(f"{len(no_random):,}", f"{len(no_random):,}".replace(",", ".")),
        f"| ước lượng cây trò chơi | ≈ 10^{tree:.1f} | tích số nước hợp lệ 121·120·…·{n - d + 1} theo độ dài ván trung bình {d} |",
        f"| tỷ lệ hòa | {draws}/{len(games)} ván chế độ B ({100 * draws / len(games):.1f}%) | hòa chỉ khi đủ 121 ô |",
        f"| lợi thế đi trước | score X = {x_score:.3f} (chế độ B); X thắng {100 * gen['outcomes']['black_win_rate']:.1f}% trong 20.000 ván sinh dữ liệu | |",
        "",
        "Tính chất: hai người chơi, quan sát đầy đủ, tất định, tổng bằng không (thắng +1, thua −1, hòa 0 theo góc nhìn người chơi).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (("moi_truong_chay.md", environment_md()), ("do_phuc_tap.md", complexity_md())):
        (OUT / name).write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
