"""Vòng lặp AlphaZero-inspired cho cấp 8 (proposal mục 4–5, issue #14).

Mỗi vòng:
  1. Self-play: mạng tốt nhất (best) tự chơi bằng PUCT + nhiễu Dirichlet.
  2. Train: mạng hiện tại (current) học từ replay buffer (`buffer_rounds` vòng gần nhất),
     tăng dữ liệu bằng 8 phép đối xứng. Loss = cross-entropy(policy) + MSE(value) + L2.
  3. Gating: current đấu best trên các khai cuộc cân bằng, đổi bên; thắng >= ngưỡng
     (mặc định 55%) thì current thành best.

Chạy tiếp khi bị ngắt: mọi thứ nằm trong `out_dir` (đặt trên Google Drive khi dùng Colab):
  state.json, best.pt, current.pt, rounds/round_XXX.pt, buffer/*.npz, train_log.jsonl.
Self-play lưu theo từng nhóm ván, nên ngắt giữa vòng chỉ mất nhóm đang chơi dở.
Chạy lại đúng lệnh cũ là làm tiếp từ chỗ dừng.

Chạy:  python -m training.alphazero configs/alphazero.yaml [--out-dir DIR] [--rounds N]
                                       [--device cuda] [--workers 6] [--max-hours 3]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from agents.network import (
    NetEvaluator,
    PolicyValueNet,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)
from agents.puct_agent import PUCTAgent
from arena.logger import JsonlLogger, hardware_info
from arena.openings import load_openings
from game.constants import ENGINE_VERSION
from game.game import CaroGame
from training.selfplay import NUM_SYMMETRIES, SelfPlayConfig, play_games, symmetry

ILLEGAL_LOGIT = -1e9
SEED_STRIDE = 1_000_003
Z95 = 1.959963984540054

DEFAULTS: dict[str, Any] = {
    "name": "alphazero",
    "out_dir": "checkpoints/alphazero",
    "seed": 2026,
    "device": "auto",
    "rounds": 8,
    "network": {"blocks": 4, "channels": 64},
    "selfplay": {"games_per_round": 100, "chunk_games": 20, "workers": 1, "simulations": 100,
                 "c_puct": 1.5, "dirichlet_alpha": 0.15, "dirichlet_eps": 0.25, "temp_moves": 10},
    "buffer_rounds": 4,
    "train": {"epochs": 2, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4},
    "gating": {"games": 40, "simulations": 100, "threshold": 0.55,
               "openings_file": "configs/openings.json"},
}


# ---------------------------------------------------------------- tiện ích

def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    return _merge(DEFAULTS, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def wilson(score: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    d = 1 + Z95 * Z95 / n
    c = (score + Z95 * Z95 / (2 * n)) / d
    h = Z95 * math.sqrt(score * (1 - score) / n + Z95 * Z95 / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- self-play

def _selfplay_cfg(cfg: dict[str, Any]) -> SelfPlayConfig:
    sp = cfg["selfplay"]
    return SelfPlayConfig(simulations=sp["simulations"], c_puct=sp["c_puct"],
                          dirichlet_alpha=sp["dirichlet_alpha"], dirichlet_eps=sp["dirichlet_eps"],
                          temp_moves=sp["temp_moves"])


def _selfplay_chunk(args: tuple[str, str, int, SelfPlayConfig, int]) -> dict[str, Any]:
    """Chạy trong tiến trình con: nạp mạng best và chơi một nhóm ván."""
    ckpt, device, n_games, sp_cfg, seed = args
    torch.set_num_threads(1)
    model, _ = load_checkpoint(ckpt, device)
    return play_games(NetEvaluator(model, device), n_games, sp_cfg, seed)


class AlphaZeroRunner:
    def __init__(self, cfg: dict[str, Any], log: bool = True) -> None:
        self.cfg = cfg
        self.out = Path(cfg["out_dir"])
        self.device = pick_device(cfg["device"])
        self.log = log
        for sub in ("rounds", "buffer"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.train_log = JsonlLogger(self.out / "train_log.jsonl")
        self.started = time.perf_counter()

    # ------------------------------------------------ trạng thái
    def _say(self, msg: str) -> None:
        if self.log:
            print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        torch.manual_seed(self.cfg["seed"])
        model = PolicyValueNet(**self.cfg["network"])
        save_checkpoint(self.out / "current.pt", model, round=0)
        save_checkpoint(self.out / "best.pt", model, round=0)
        state = {"round": 1, "phase": "selfplay", "games_done": 0, "best_round": 0, "rounds_info": {}}
        _atomic_json(self.state_path, state)
        JsonlLogger(self.out / "meta.jsonl").write({
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine_version": ENGINE_VERSION, "config": self.cfg, "hardware": hardware_info(),
            "device": str(self.device), "torch": torch.__version__,
        })
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        _atomic_json(self.state_path, state)

    def _time_up(self, max_hours: float | None) -> bool:
        return max_hours is not None and time.perf_counter() - self.started > max_hours * 3600

    # ------------------------------------------------ 1. self-play
    def selfplay(self, state: dict[str, Any], max_hours: float | None) -> bool:
        """Trả về True nếu chơi đủ ván của vòng; False nếu dừng vì hết giờ."""
        sp = self.cfg["selfplay"]
        rnd = state["round"]
        total, chunk = sp["games_per_round"], sp["chunk_games"]
        sp_cfg = _selfplay_cfg(self.cfg)
        best = str(self.out / "best.pt")
        workers = int(sp["workers"]) if self.device.type == "cpu" else 1
        t0 = time.perf_counter()

        def jobs_from(done: int) -> list[tuple[int, int]]:
            out, k = [], done
            while k < total:
                n = min(chunk, total - k)
                out.append((k, n))
                k += n
            return out

        def save(start: int, n: int, data: dict[str, Any]) -> None:
            part = self.out / "buffer" / f"r{rnd:03d}_g{start:04d}_{n:03d}.npz"
            _atomic_npz(part, **data)

        jobs = jobs_from(state["games_done"])
        seed_of = lambda start: self.cfg["seed"] * SEED_STRIDE + rnd * 10_000 + start  # noqa: E731
        if workers <= 1:
            model, _ = load_checkpoint(best, self.device)
            evaluator = NetEvaluator(model, self.device)
            for start, n in jobs:
                if self._time_up(max_hours):
                    return False
                save(start, n, play_games(evaluator, n, sp_cfg, seed_of(start)))
                state["games_done"] = start + n
                self.save_state(state)
                self._say(f"vòng {rnd}: self-play {state['games_done']}/{total} ván")
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                for batch_start in range(0, len(jobs), workers):
                    if self._time_up(max_hours):
                        return False
                    batch = jobs[batch_start:batch_start + workers]
                    args = [(best, "cpu", n, sp_cfg, seed_of(s)) for s, n in batch]
                    for (start, n), data in zip(batch, pool.map(_selfplay_chunk, args)):
                        save(start, n, data)
                    state["games_done"] = batch[-1][0] + batch[-1][1]
                    self.save_state(state)
                    self._say(f"vòng {rnd}: self-play {state['games_done']}/{total} ván")
        info = state["rounds_info"].setdefault(str(rnd), {})
        info["selfplay_s"] = info.get("selfplay_s", 0.0) + time.perf_counter() - t0
        return True

    # ------------------------------------------------ 2. train
    def load_buffer(self, rnd: int) -> dict[str, np.ndarray]:
        first = max(1, rnd - self.cfg["buffer_rounds"] + 1)
        parts = sorted(p for p in (self.out / "buffer").glob("r*_g*.npz")
                       if first <= int(p.name[1:4]) <= rnd and ".tmp" not in p.name)
        data: dict[str, list[np.ndarray]] = {"planes": [], "pi": [], "z": []}
        for p in parts:
            with np.load(p) as f:
                for k in data:
                    data[k].append(f[k])
        return {k: np.concatenate(v) for k, v in data.items()}

    def train(self, state: dict[str, Any]) -> dict[str, float]:
        rnd = state["round"]
        tr = self.cfg["train"]
        t0 = time.perf_counter()
        buf = self.load_buffer(rnd)
        payload = torch.load(self.out / "current.pt", map_location=self.device, weights_only=False)
        model = PolicyValueNet(**payload["config"]).to(self.device)
        model.load_state_dict(payload["model_state"])
        opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
        if payload.get("optimizer_state"):
            opt.load_state_dict(payload["optimizer_state"])

        rng = np.random.default_rng(self.cfg["seed"] * SEED_STRIDE + rnd)
        n = len(buf["z"])
        bs = tr["batch_size"]
        model.train()
        p_losses, v_losses = [], []
        for _ in range(tr["epochs"]):
            order = rng.permutation(n)
            for b in range(0, n, bs):
                idx = order[b:b + bs]
                planes, pi = buf["planes"][idx].astype(np.float32), buf["pi"][idx]
                k = int(rng.integers(NUM_SYMMETRIES))  # một phép đối xứng ngẫu nhiên cho cả batch
                planes, pi = symmetry(planes, pi, k)
                x = torch.from_numpy(planes).to(self.device)
                target_pi = torch.from_numpy(pi).to(self.device)
                target_z = torch.from_numpy(buf["z"][idx]).to(self.device)
                occupied = (x[:, 0] + x[:, 1]).flatten(1) > 0
                logits, value = model(x)
                logits = logits.masked_fill(occupied, ILLEGAL_LOGIT)
                p_loss = -(target_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                v_loss = F.mse_loss(value, target_z)
                opt.zero_grad()
                (p_loss + v_loss).backward()
                opt.step()
                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
        model.eval()
        extra = {"round": rnd, "optimizer_state": opt.state_dict()}
        save_checkpoint(self.out / "rounds" / f"round_{rnd:03d}.pt", model, round=rnd)
        save_checkpoint(self.out / "current.pt", model, **extra)
        return {
            "positions": int(n),
            "policy_loss": float(np.mean(p_losses[-max(1, len(p_losses) // tr["epochs"]):])),
            "value_loss": float(np.mean(v_losses[-max(1, len(v_losses) // tr["epochs"]):])),
            "train_s": time.perf_counter() - t0,
        }

    # ------------------------------------------------ 3. gating
    def gate(self, state: dict[str, Any]) -> dict[str, Any]:
        g = self.cfg["gating"]
        t0 = time.perf_counter()
        cand_model, _ = load_checkpoint(self.out / "current.pt", self.device)
        best_model, _ = load_checkpoint(self.out / "best.pt", self.device)
        cand = PUCTAgent(evaluator=NetEvaluator(cand_model, self.device), simulations=g["simulations"])
        best = PUCTAgent(evaluator=NetEvaluator(best_model, self.device), simulations=g["simulations"])
        openings = load_openings(g["openings_file"])[: (g["games"] + 1) // 2]
        points, games = 0.0, 0
        for op in openings:
            for cand_is_x in (True, False):
                if games >= g["games"]:
                    break
                game = CaroGame()
                for m in op["moves"]:
                    game.step(m)
                while not game.is_terminal():
                    me = cand if (game.current_player == 1) == cand_is_x else best
                    game.step(me.select_action(game))
                r = game.result()
                points += 0.5 if r == 0 else float((r == 1) == cand_is_x)
                games += 1
        score = points / games
        accepted = score >= g["threshold"]
        if accepted:
            shutil.copyfile(self.out / "current.pt", self.out / "best.tmp")
            os.replace(self.out / "best.tmp", self.out / "best.pt")
        lo, hi = wilson(score, games)
        return {"gate_games": games, "gate_score": score, "gate_ci95": [lo, hi],
                "accepted": accepted, "gate_s": time.perf_counter() - t0}

    # ------------------------------------------------ vòng lặp
    def run(self, rounds: int, max_hours: float | None = None) -> dict[str, Any]:
        state = self.load_state()
        self._say(f"bắt đầu ở vòng {state['round']} ({state['phase']}), thiết bị {self.device}, "
                  f"thư mục {self.out}")
        while state["round"] <= rounds:
            rnd = state["round"]
            info = state["rounds_info"].setdefault(str(rnd), {})
            if state["phase"] == "selfplay":
                if not self.selfplay(state, max_hours):
                    self._say("hết thời gian cho phép, dừng; chạy lại lệnh cũ để làm tiếp")
                    return state
                state["phase"] = "train"
                self.save_state(state)
            if self._time_up(max_hours):
                self._say("hết thời gian cho phép, dừng; chạy lại lệnh cũ để làm tiếp")
                return state
            if state["phase"] == "train":
                info.update(self.train(state))
                state["phase"] = "gate"
                self.save_state(state)
                self._say(f"vòng {rnd}: train xong, policy_loss {info['policy_loss']:.3f}, "
                          f"value_loss {info['value_loss']:.3f}")
            if state["phase"] == "gate":
                info.update(self.gate(state))
                if info["accepted"]:
                    state["best_round"] = rnd
                stats = self._selfplay_stats(rnd)
                self.train_log.write({"round": rnd, "best_round": state["best_round"],
                                      "device": str(self.device), **stats, **info})
                self._say(f"vòng {rnd}: gating score {info['gate_score']:.2f} -> "
                          f"{'NHẬN' if info['accepted'] else 'giữ best cũ'}")
                state.update({"round": rnd + 1, "phase": "selfplay", "games_done": 0})
                self.save_state(state)
        self._say(f"xong {rounds} vòng; best là vòng {state['best_round']}")
        return state

    def _selfplay_stats(self, rnd: int) -> dict[str, Any]:
        results, lengths = [], []
        for p in sorted((self.out / "buffer").glob(f"r{rnd:03d}_g*.npz")):
            with np.load(p) as f:
                results.append(f["results"])
                lengths.append(f["lengths"])
        if not results:
            return {}
        r, n = np.concatenate(results), np.concatenate(lengths)
        return {"selfplay_games": int(len(r)), "selfplay_x_win": float((r == 1).mean()),
                "selfplay_o_win": float((r == -1).mean()), "selfplay_draw": float((r == 0).mean()),
                "selfplay_mean_len": float(n.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vòng lặp AlphaZero-inspired (cấp 8).")
    parser.add_argument("config")
    parser.add_argument("--out-dir", default=None, help="Ghi đè out_dir (ví dụ thư mục trên Google Drive)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto / cpu / cuda / mps")
    parser.add_argument("--workers", type=int, default=None, help="Số tiến trình self-play khi chạy CPU")
    parser.add_argument("--max-hours", type=float, default=None, help="Tự dừng an toàn sau số giờ này")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.device:
        cfg["device"] = args.device
    if args.workers:
        cfg["selfplay"]["workers"] = args.workers
    rounds = args.rounds or cfg["rounds"]
    AlphaZeroRunner(cfg).run(rounds, args.max_hours)


if __name__ == "__main__":
    main()
