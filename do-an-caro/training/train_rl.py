"""Huấn luyện Cấp 6: RL Self-Play Actor-Critic khởi tạo từ mạng Cấp 5 (Issue #11).

Tuân thủ Proposal mục 4–5 và Issue #11:
- Khởi tạo từ checkpoint Cấp 5 (`checkpoints/level5/best.pt`, `PolicyValueNet(blocks=6, channels=64)`).
- Policy Gradient có baseline (Actor-Critic):
  - Phần thưởng cuối ván `z` (+1 thắng, -1 thua, 0 hòa) theo góc nhìn người chơi RL; 0 ở các bước trung gian.
  - Advantage `A_t = z - V(s_t).detach()`, loss = `-mean(A_t * log_pi(a_t|s_t)) + c_v * MSE(V, z) - c_ent * H(pi)`.
  - Mask ô bận (`-1e9`) trước softmax/log_softmax và lấy mẫu theo nhiệt độ `tau` khi khám phá.
- Pool checkpoint làm đối thủ: luôn giữ bản neo Cấp 5 (SL) cùng các checkpoint lịch sử gần nhất
  để tự chơi luân phiên X/O và tránh quên thảm họa (catastrophic forgetting).
- Đánh giá định kỳ với mạng Cấp 5 trên khai cuộc cân bằng (cho đường cong học tập RQ3),
  lưu `rounds/round_XXX.pt`, `best.pt`, `latest.pt` (tự động chạy tiếp khi bị ngắt).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from agents.network import (
    PolicyValueNet,
    encode_game,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)
from analysis.arena_stats import wilson_interval
from arena.openings import load_openings
from game.constants import BLACK, DEFAULT_BOARD_SIZE, ENGINE_VERSION, RULE_VIETNAM, WHITE
from game.game import CaroGame
from training.selfplay import NUM_SYMMETRIES
from training.train_supervised_policy_value_network import (
    ILLEGAL_LOGIT,
    _capture_rng_state,
    _restore_rng_state,
    mask_illegal_logits,
    set_seed,
)

DEFAULT_RL_OUT_DIR = "checkpoints/level6"
DEFAULT_SL_INIT = "checkpoints/level5/best.pt"


def batch_symmetry(
    planes: np.ndarray,
    actions: np.ndarray,
    k: int,
    size: int = DEFAULT_BOARD_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Áp dụng phép đối xứng thứ k (0..7) lên batch `planes` (N,4,S,S) và `actions` (N,)."""
    rot = k % 4
    p = np.rot90(planes, rot, axes=(-2, -1)) if rot else planes
    r, c = np.divmod(actions, size)
    if rot == 1:
        r, c = size - 1 - c, r
    elif rot == 2:
        r, c = size - 1 - r, size - 1 - c
    elif rot == 3:
        r, c = c, size - 1 - r
    if k >= 4:
        p = np.flip(p, axis=-1)
        c = size - 1 - c
    return np.ascontiguousarray(p), (r * size + c).astype(np.int64)


def _encode_games_batch(games: list[CaroGame]) -> tuple[np.ndarray, np.ndarray]:
    planes_list, masks_list = [], []
    for g in games:
        p, m = encode_game(g)
        planes_list.append(p)
        masks_list.append(m)
    return np.stack(planes_list), np.stack(masks_list)


@torch.no_grad()
def sample_batched_actions(
    model: PolicyValueNet,
    games: list[CaroGame],
    temperature: float,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Chọn nước đi theo lô cho danh sách `games` (trả về `actions` (N,) và `planes` uint8 (N,4,S,S))."""
    model.eval()
    planes, masks = _encode_games_batch(games)
    x = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).to(device)
    logits, _ = model(x)
    mask_t = torch.from_numpy(masks).to(device)
    masked_logits = logits.masked_fill(~mask_t, ILLEGAL_LOGIT)

    if temperature <= 1e-6:
        actions = masked_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    else:
        probs = F.softmax(masked_logits / float(temperature), dim=-1).float().cpu().numpy()
        probs = np.where(masks, probs, 0.0).astype(np.float64)
        cdf = np.cumsum(probs, axis=1)
        totals = cdf[:, -1:]
        # Đảm bảo không bao giờ chọn ô có xác suất 0 ngay cả khi rng.random() == 0.0
        u = rng.random((len(probs), 1)) * totals
        actions = (cdf > u).argmax(axis=1).astype(np.int64)

    return actions, planes.astype(np.uint8)


def _play_random_opening(
    game: CaroGame,
    min_moves: int,
    max_moves: int,
    center_radius: int,
    rng: np.random.Generator,
) -> None:
    """Đi ngẫu nhiên `min_moves..max_moves` nước quanh tâm để đa dạng thế cờ self-play."""
    if max_moves <= 0:
        return
    n_open = int(rng.integers(min_moves, max_moves + 1))
    center = game.size // 2
    lo = max(0, center - center_radius)
    hi = min(game.size - 1, center + center_radius)
    candidates = [r * game.size + c for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    chosen = rng.choice(candidates, size=min(n_open, len(candidates)), replace=False)
    for act in chosen:
        if game.is_terminal():
            break
        game.step(int(act))


def collect_selfplay_batch(
    rl_model: PolicyValueNet,
    opp_model: PolicyValueNet,
    n_games: int,
    cfg: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Chơi song song `n_games` ván (nửa cầm X, nửa cầm O) và thu thập trajectory của tác tử RL."""
    size = int(cfg.get("board_size", DEFAULT_BOARD_SIZE))
    rule = str(cfg.get("rule", RULE_VIETNAM))
    temp = float(cfg.get("temperature", 1.0))
    opp_temp = float(cfg.get("opponent_temperature", 1.0))
    min_open = int(cfg.get("opening_min_moves", 2))
    max_open = int(cfg.get("opening_max_moves", 4))
    center_radius = int(cfg.get("center_radius", 2))

    games: list[CaroGame] = []
    rl_players: list[int] = []
    for i in range(n_games):
        g = CaroGame(size=size, rule=rule)
        _play_random_opening(g, min_open, max_open, center_radius, rng)
        games.append(g)
        rl_players.append(BLACK if i % 2 == 0 else WHITE)

    trajectories: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(n_games)]
    active = [i for i, g in enumerate(games) if not g.is_terminal()]

    while active:
        rl_idx = [i for i in active if games[i].current_player == rl_players[i]]
        opp_idx = [i for i in active if games[i].current_player != rl_players[i]]

        if rl_idx:
            rl_actions, rl_planes = sample_batched_actions(
                rl_model, [games[i] for i in rl_idx], temp, device, rng
            )
            for k, g_i in enumerate(rl_idx):
                act = int(rl_actions[k])
                trajectories[g_i].append((rl_planes[k], act))
                games[g_i].step(act)

        if opp_idx:
            opp_actions, _ = sample_batched_actions(
                opp_model, [games[i] for i in opp_idx], opp_temp, device, rng
            )
            for k, g_i in enumerate(opp_idx):
                games[g_i].step(int(opp_actions[k]))

        active = [i for i in active if not games[i].is_terminal()]

    planes_out: list[np.ndarray] = []
    actions_out: list[int] = []
    returns_out: list[float] = []
    wins = draws = losses = total_len = 0

    for g, rl_p, traj in zip(games, rl_players, trajectories):
        res = int(g.result())
        total_len += g.move_count
        if res == 0:
            z = 0.0
            draws += 1
        elif res == rl_p:
            z = 1.0
            wins += 1
        else:
            z = -1.0
            losses += 1
        for p, a in traj:
            planes_out.append(p)
            actions_out.append(a)
            returns_out.append(z)

    return {
        "planes": np.stack(planes_out) if planes_out else np.empty((0, 4, size, size), dtype=np.uint8),
        "actions": np.asarray(actions_out, dtype=np.int64),
        "returns": np.asarray(returns_out, dtype=np.float32),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": n_games,
        "mean_reward": (wins - losses) / max(1, n_games),
        "mean_length": total_len / max(1, n_games),
    }


def actor_critic_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    planes: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    sl_model: PolicyValueNet | None = None,
    value_only: bool = False,
) -> dict[str, float]:
    """Cập nhật Actor-Critic (Policy Gradient có baseline Value head + phạt KL) trên một rollout batch."""
    n = len(actions)
    if n == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "loss": 0.0}

    minibatch_size = int(cfg.get("minibatch_size", 256))
    value_coef = float(cfg.get("value_coef", 0.5))
    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    kl_coef = float(cfg.get("kl_coef", 0.0))
    normalize_adv = bool(cfg.get("normalize_advantage", True))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    augment = bool(cfg.get("augment", True))
    size = int(cfg.get("board_size", DEFAULT_BOARD_SIZE))

    if sl_model is not None:
        sl_model.eval()

    model.train()
    frozen_modules = (model.stem, model.blocks, model.policy_conv, model.policy_fc)
    if value_only:
        # Đóng băng backbone và đầu policy (kể cả BatchNorm) khi khởi ấm riêng đầu value
        for mod in frozen_modules:
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)

    order = rng.permutation(n)
    p_losses, v_losses, ents, kls, tot_losses = [], [], [], [], []

    try:
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            p_b = planes[idx].astype(np.float32)
            a_b = actions[idx]
            r_b = returns[idx]

            if augment:
                sym_k = int(rng.integers(NUM_SYMMETRIES))
                p_b, a_b = batch_symmetry(p_b, a_b, sym_k, size=size)

            x = torch.from_numpy(p_b).to(device)
            act_t = torch.from_numpy(a_b).to(device)
            ret_t = torch.from_numpy(r_b).to(device)

            logits, values = model(x)
            logits = mask_illegal_logits(logits, x)

            legal_mask = (x[:, 0] + x[:, 1]).flatten(1) == 0
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            chosen_log_probs = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)

            advantages = ret_t - values.detach()
            if normalize_adv and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * chosen_log_probs).mean()
            value_loss = F.mse_loss(values, ret_t)
            entropy = -(probs * log_probs).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()

            # Phạt KL(pi || pi_SL) để policy không trôi xa mạng Cấp 5
            kl = torch.zeros((), device=x.device)
            if kl_coef > 0 and sl_model is not None:
                with torch.no_grad():
                    sl_logits, _ = sl_model(x)
                    sl_log_probs = F.log_softmax(mask_illegal_logits(sl_logits, x), dim=-1)
                kl = (probs * (log_probs - sl_log_probs)).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()

            if value_only:
                loss = value_coef * value_loss
            else:
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                trainable = [p for p in model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()

            p_losses.append(0.0 if value_only else float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
            ents.append(float(entropy.item()))
            kls.append(float(kl.item()))
            tot_losses.append(float(loss.item()))
    finally:
        if value_only:
            for mod in frozen_modules:
                for p in mod.parameters():
                    p.requires_grad_(True)
        model.eval()

    return {
        "policy_loss": float(np.mean(p_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": float(np.mean(ents)),
        "kl": float(np.mean(kls)),
        "loss": float(np.mean(tot_losses)),
    }


class OpponentPool:
    """Quản lý pool các checkpoint cũ (luôn ghim bản neo SL để tránh quên)."""

    def __init__(self, pool_dir: Path, anchor_path: Path, max_size: int = 20) -> None:
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.anchor_path = self.pool_dir / "anchor_sl.pt"
        if not self.anchor_path.exists():
            shutil.copyfile(anchor_path, self.anchor_path)
        self.max_size = max(1, int(max_size))
        self.snapshots: list[Path] = sorted(self.pool_dir.glob("round_*.pt"))
        self._cache: dict[str, PolicyValueNet] = {}

    def add_snapshot(self, round_ckpt: Path, rnd: int) -> None:
        if self.max_size <= 1:
            return
        dst = self.pool_dir / f"round_{rnd:03d}.pt"
        shutil.copyfile(round_ckpt, dst)
        if dst not in self.snapshots:
            self.snapshots.append(dst)
        max_hist = self.max_size - 1
        while len(self.snapshots) > max_hist:
            oldest = self.snapshots.pop(0)
            self._cache.pop(str(oldest), None)
            if oldest.exists():
                oldest.unlink()

    def paths(self) -> list[Path]:
        return [self.anchor_path, *self.snapshots]

    def sample_model(self, rng: np.random.Generator, device: torch.device) -> tuple[PolicyValueNet, Path]:
        candidates = self.paths()
        chosen = candidates[int(rng.integers(len(candidates)))]
        key = str(chosen)
        if key not in self._cache:
            model, _ = load_checkpoint(chosen, device)
            self._cache[key] = model
        return self._cache[key], chosen


@torch.no_grad()
def evaluate_vs_baseline(
    rl_model: PolicyValueNet,
    baseline_model: PolicyValueNet,
    openings: list[dict[str, Any]],
    eval_games: int,
    size: int,
    rule: str,
    device: torch.device,
) -> dict[str, Any]:
    """Đánh giá tất định (argmax policy) giữa RL và SL trên các khai cuộc cân bằng đổi bên."""
    rl_model.eval()
    baseline_model.eval()
    rng = np.random.default_rng(0)  # không dùng khi temp=0.0 nhưng cần đúng chữ ký

    games: list[CaroGame] = []
    rl_players: list[int] = []
    n_pairs = max(1, (eval_games + 1) // 2)
    op_list = openings if openings else [{"moves": []} for _ in range(n_pairs)]

    for i in range(eval_games):
        op = op_list[(i // 2) % len(op_list)]
        rl_is_x = (i % 2 == 0)
        g = CaroGame(size=size, rule=rule)
        for m in op.get("moves", []):
            if not g.is_terminal():
                g.step(int(m))
        games.append(g)
        rl_players.append(BLACK if rl_is_x else WHITE)

    active = [i for i, g in enumerate(games) if not g.is_terminal()]
    while active:
        rl_idx = [i for i in active if games[i].current_player == rl_players[i]]
        opp_idx = [i for i in active if games[i].current_player != rl_players[i]]

        if rl_idx:
            acts, _ = sample_batched_actions(rl_model, [games[i] for i in rl_idx], 0.0, device, rng)
            for k, g_i in enumerate(rl_idx):
                games[g_i].step(int(acts[k]))
        if opp_idx:
            acts, _ = sample_batched_actions(baseline_model, [games[i] for i in opp_idx], 0.0, device, rng)
            for k, g_i in enumerate(opp_idx):
                games[g_i].step(int(acts[k]))

        active = [i for i in active if not games[i].is_terminal()]

    wins = draws = losses = 0
    for g, rl_p in zip(games, rl_players):
        r = int(g.result())
        if r == 0:
            draws += 1
        elif r == rl_p:
            wins += 1
        else:
            losses += 1
    n = len(games)
    score = (wins + 0.5 * draws) / max(1, n)
    lo, hi = wilson_interval(score, n)
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _sync_rl_log(log_path: Path, max_round: int) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    by_round: dict[int, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rnd = int(rec.get("round", -1))
        if 0 <= rnd <= max_round:
            by_round[rnd] = rec
    history = [by_round[r] for r in sorted(by_round)]
    tmp = log_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history), encoding="utf-8")
    os.replace(tmp, log_path)
    return history


def train_rl(
    config: dict[str, Any],
    out_dir: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    no_resume: bool = False,
    device_name: str | None = None,
    max_hours: float | None = None,
) -> dict[str, Any]:
    """Chạy vòng lặp huấn luyện Actor-Critic RL self-play cho Cấp 6."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    target_out_dir = Path(out_dir or config.get("out_dir", DEFAULT_RL_OUT_DIR))
    target_out_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = target_out_dir / "rounds"

    dev_name = device_name or config.get("device", "auto")
    device = pick_device(dev_name)

    init_ckpt_path = Path(init_checkpoint or config.get("init_checkpoint", DEFAULT_SL_INIT))
    total_episodes = int(config.get("episodes", 20000))
    episodes_per_round = int(config.get("episodes_per_round", 1000))
    num_rounds = max(1, math.ceil(total_episodes / episodes_per_round))
    rollout_batch = int(config.get("rollout_batch_games", 64))
    lr = float(config.get("lr", 1e-5))
    weight_decay = float(config.get("weight_decay", 1e-4))
    value_warmup_rounds = max(0, int(config.get("value_warmup_rounds", 0)))
    size = int(config.get("board_size", DEFAULT_BOARD_SIZE))
    rule = str(config.get("rule", RULE_VIETNAM))
    max_pool_size = int(config.get("max_pool_size", 20))
    eval_games = int(config.get("eval_games", 100))

    openings_file = config.get("openings_file")
    openings = load_openings(openings_file) if openings_file and Path(openings_file).exists() else []

    latest_path = target_out_dir / "latest.pt"
    log_path = target_out_dir / "train_log.jsonl"

    if no_resume:
        resume_path = None
    elif resume is not None:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint resume: {resume_path}")
    elif latest_path.exists():
        resume_path = latest_path
    else:
        resume_path = None

    print("=" * 70)
    print("Huấn luyện Cấp 6 – RL Self-Play Actor-Critic")
    print(f"Thiết bị: {device} | Engine: v{ENGINE_VERSION} | Seed: {seed}")
    print(f"Mục tiêu: {total_episodes:,} episodes ({num_rounds} vòng × {episodes_per_round} ep) | Out: {target_out_dir}")
    print("=" * 70)

    if resume_path is not None:
        rounds_dir.mkdir(parents=True, exist_ok=True)
        print(f"Phát hiện checkpoint {resume_path}, tự động tiếp tục...")
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model = PolicyValueNet(**payload["config"]).to(device)
        model.load_state_dict(payload["model_state"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_round = int(payload.get("round", 0)) + 1
        episodes_done = int(payload.get("episodes_done", 0))
        best_score = float(payload.get("best_score_vs_sl", 0.5))
        best_round = int(payload.get("best_round", 0))
        _restore_rng_state(payload.get("rng_state"), device)
        if payload.get("np_rng_bit_state"):
            rng.bit_generator.state = payload["np_rng_bit_state"]
        history = _sync_rl_log(log_path, start_round - 1)
        anchor_src = target_out_dir / "pool" / "anchor_sl.pt"
        if not anchor_src.exists():
            anchor_src = init_ckpt_path
        sl_baseline, _ = load_checkpoint(anchor_src, device)
        pool = OpponentPool(target_out_dir / "pool", anchor_src, max_size=max_pool_size)
        print(f"  Tiếp tục từ vòng {start_round} (đã chạy {episodes_done:,} ep) | best={best_score:.3f} (vòng {best_round})")
    else:
        if not init_ckpt_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint Cấp 5 (SL) tại: {init_ckpt_path}.\n"
                "Cấp 6 bắt buộc khởi tạo từ mạng Cấp 5 theo Proposal mục 4–5."
            )
        # Dọn sạch pool, rounds và checkpoint cũ khi chạy mới / --no-resume để không lẫn bản neo SL cũ
        for stale_dir in (target_out_dir / "pool", rounds_dir):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        for stale_file in (log_path, latest_path, target_out_dir / "best.pt", target_out_dir / "summary.json"):
            if stale_file.exists():
                stale_file.unlink()
        rounds_dir.mkdir(parents=True, exist_ok=True)

        model, sl_meta = load_checkpoint(init_ckpt_path, device)
        sl_baseline, _ = load_checkpoint(init_ckpt_path, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        pool = OpponentPool(target_out_dir / "pool", init_ckpt_path, max_size=max_pool_size)

        # Lưu điểm gốc vòng 0 (chính là mạng Cấp 5) để vẽ đường cong học tập RQ3
        r0_path = rounds_dir / "round_000.pt"
        save_checkpoint(r0_path, model, epoch=0, val_metrics={"score_vs_sl": 0.5}, seed=seed, init_from=str(init_ckpt_path))
        save_checkpoint(target_out_dir / "best.pt", model, epoch=0, val_metrics={"score_vs_sl": 0.5}, seed=seed)

        r0_eval = evaluate_vs_baseline(model, sl_baseline, openings, eval_games, size, rule, device)
        best_score = float(r0_eval["score"])
        best_round = 0
        start_round = 1
        episodes_done = 0

        r0_rec = {
            "round": 0,
            "episodes": 0,
            "policy_loss": None,
            "value_loss": None,
            "entropy": None,
            "kl": None,
            "mean_reward": 0.0,
            "mean_game_len": 0.0,
            "score_vs_sl": r0_eval["score"],
            "ci95_vs_sl": r0_eval["ci95"],
            "wins_vs_sl": r0_eval["wins"],
            "draws_vs_sl": r0_eval["draws"],
            "losses_vs_sl": r0_eval["losses"],
            "is_best": True,
            "pool_size": len(pool.paths()),
            "round_s": 0.0,
        }
        history = [r0_rec]
        log_path.write_text(json.dumps(r0_rec, ensure_ascii=False) + "\n", encoding="utf-8")
        save_checkpoint(
            latest_path, model, epoch=0, val_metrics=r0_eval, seed=seed,
            round=0, episodes_done=0, best_score_vs_sl=best_score, best_round=0,
            optimizer_state=optimizer.state_dict(), rng_state=_capture_rng_state(device),
            np_rng_bit_state=rng.bit_generator.state,
        )
        print(f"Khởi tạo từ Cấp 5 ({init_ckpt_path}, {sum(p.numel() for p in model.parameters()):,} tham số). Vòng 0 score vs SL: {best_score:.3f}")

    cfg_saved = dict(config)
    cfg_saved.update({
        "out_dir": str(target_out_dir),
        "init_checkpoint": str(init_ckpt_path),
        "device": str(device),
        "engine_version": ENGINE_VERSION,
    })
    (target_out_dir / "config_used.yaml").write_text(yaml.dump(cfg_saved, allow_unicode=True), encoding="utf-8")

    t_start = time.perf_counter()
    if start_round > num_rounds:
        print(f"Đã hoàn tất đủ {num_rounds} vòng ({episodes_done:,} episodes). Giữ nguyên best.pt.")
    else:
        print(f"{'Vòng':>4} | {'Episodes':>8} | {'P-Loss':>8} | {'V-Loss':>8} | {'Ent':>6} | {'KL':>7} | {'Reward':>7} | {'Score SL':>8} | {'CI95':>15} | {'Pool':>4} | {'Time':>6}")
        print("-" * 110)

        for rnd in range(start_round, num_rounds + 1):
            if max_hours is not None and (time.perf_counter() - t_start) > max_hours * 3600:
                print(f"[Time Limit] Hết giới hạn {max_hours} giờ, dừng an toàn trước vòng {rnd}. Chạy lại để tiếp tục.")
                break

            r_start = time.perf_counter()
            ep_target = min(episodes_per_round, total_episodes - episodes_done)
            if ep_target <= 0:
                break

            ep_in_round = 0
            is_value_warmup = rnd <= value_warmup_rounds
            p_losses, v_losses, ents, kls, rewards, lengths = [], [], [], [], [], []
            while ep_in_round < ep_target:
                n_batch = min(rollout_batch, ep_target - ep_in_round)
                opp_model, _ = pool.sample_model(rng, device)
                rollout = collect_selfplay_batch(model, opp_model, n_batch, config, device, rng)
                upd = actor_critic_update(
                    model, optimizer,
                    rollout["planes"], rollout["actions"], rollout["returns"],
                    config, device, rng,
                    sl_model=sl_baseline,
                    value_only=is_value_warmup,
                )
                ep_in_round += n_batch
                episodes_done += n_batch
                p_losses.append(upd["policy_loss"])
                v_losses.append(upd["value_loss"])
                ents.append(upd["entropy"])
                kls.append(upd["kl"])
                rewards.append(rollout["mean_reward"])
                lengths.append(rollout["mean_length"])

            eval_res = evaluate_vs_baseline(model, sl_baseline, openings, eval_games, size, rule, device)
            score_vs_sl = float(eval_res["score"])
            is_best = score_vs_sl > best_score

            round_ckpt = rounds_dir / f"round_{rnd:03d}.pt"
            save_checkpoint(
                round_ckpt, model, epoch=rnd,
                val_metrics={"score_vs_sl": score_vs_sl, "ci95_vs_sl": eval_res["ci95"]},
                seed=seed, episodes=episodes_done,
            )
            pool.add_snapshot(round_ckpt, rnd)

            if is_best:
                best_score = score_vs_sl
                best_round = rnd
                save_checkpoint(
                    target_out_dir / "best.pt", model, epoch=rnd,
                    val_metrics={"score_vs_sl": score_vs_sl, "ci95_vs_sl": eval_res["ci95"]},
                    seed=seed, episodes=episodes_done,
                )

            save_checkpoint(
                latest_path, model, epoch=rnd, val_metrics=eval_res, seed=seed,
                round=rnd, episodes_done=episodes_done,
                best_score_vs_sl=best_score, best_round=best_round,
                optimizer_state=optimizer.state_dict(),
                rng_state=_capture_rng_state(device),
                np_rng_bit_state=rng.bit_generator.state,
            )

            r_sec = time.perf_counter() - r_start
            rec = {
                "round": rnd,
                "episodes": episodes_done,
                "policy_loss": round(float(np.mean(p_losses)), 5),
                "value_loss": round(float(np.mean(v_losses)), 5),
                "entropy": round(float(np.mean(ents)), 4),
                "kl": round(float(np.mean(kls)), 5),
                "mean_reward": round(float(np.mean(rewards)), 4),
                "mean_game_len": round(float(np.mean(lengths)), 2),
                "score_vs_sl": eval_res["score"],
                "ci95_vs_sl": eval_res["ci95"],
                "wins_vs_sl": eval_res["wins"],
                "draws_vs_sl": eval_res["draws"],
                "losses_vs_sl": eval_res["losses"],
                "is_best": is_best,
                "pool_size": len(pool.paths()),
                "round_s": round(r_sec, 2),
            }
            history.append(rec)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            lo, hi = eval_res["ci95"]
            mark = "*" if is_best else " "
            print(
                f"{rnd:4d} | {episodes_done:8d} | {rec['policy_loss']:8.4f} | {rec['value_loss']:8.4f} | "
                f"{rec['entropy']:6.3f} | {rec['kl']:7.4f} | {rec['mean_reward']:+7.3f} | {score_vs_sl:7.3f}{mark} | "
                f"[{lo:.3f}, {hi:.3f}] | {rec['pool_size']:4d} | {r_sec:5.1f}s"
            )

    total_time = time.perf_counter() - t_start
    summary = {
        "episodes_done": episodes_done,
        "rounds_done": len(history) - 1 if history else 0,
        "best_round": best_round,
        "best_score_vs_sl": round(best_score, 4),
        "total_time_seconds": round(total_time, 2),
        "best_checkpoint": str(target_out_dir / "best.pt"),
    }
    tmp_sum = target_out_dir / "summary.tmp"
    tmp_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_sum, target_out_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện Cấp 6 RL Self-Play Actor-Critic (Issue #11).")
    parser.add_argument("config", nargs="?", default=None, help="File YAML cấu hình.")
    parser.add_argument("--init-checkpoint", dest="init_checkpoint", help="Checkpoint Cấp 5 khởi tạo.")
    parser.add_argument("--out-dir", dest="out_dir", help="Thư mục lưu checkpoints và logs.")
    parser.add_argument("--episodes", type=int, help="Tổng số ván self-play (20.000–50.000).")
    parser.add_argument("--episodes-per-round", dest="episodes_per_round", type=int, help="Số ván mỗi vòng đánh giá.")
    parser.add_argument("--rollout-batch", dest="rollout_batch_games", type=int, help="Số ván chơi song song mỗi lô.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--temperature", type=float, help="Nhiệt độ lấy mẫu khám phá.")
    parser.add_argument("--opponent-temperature", dest="opponent_temperature", type=float, help="Nhiệt độ lấy mẫu đối thủ trong pool.")
    parser.add_argument("--kl-coef", dest="kl_coef", type=float, help="Hệ số phạt KL(pi || pi_SL).")
    parser.add_argument(
        "--normalize-advantage",
        dest="normalize_advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Chuẩn hóa advantage theo minibatch.",
    )
    parser.add_argument("--value-warmup-rounds", dest="value_warmup_rounds", type=int, help="Số vòng đầu chỉ train đầu value.")
    parser.add_argument("--device", help="Thiết bị ('cuda', 'cpu', 'auto').")
    parser.add_argument("--seed", type=int, help="Seed ngẫu nhiên.")
    parser.add_argument("--max-hours", dest="max_hours", type=float, help="Dừng an toàn sau số giờ.")
    parser.add_argument("--resume", help="Đường dẫn checkpoint chỉ định để chạy tiếp.")
    parser.add_argument("--no-resume", action="store_true", help="Huấn luyện lại từ đầu từ checkpoint Cấp 5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Lỗi: Không tìm thấy file config {args.config}", file=sys.stderr)
            sys.exit(1)
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    for k, v in vars(args).items():
        if v is not None and k not in ("config", "resume", "no_resume", "max_hours"):
            config[k] = v

    train_rl(
        config=config,
        out_dir=args.out_dir,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        no_resume=args.no_resume,
        device_name=args.device,
        max_hours=args.max_hours,
    )


if __name__ == "__main__":
    main()
