"""Kiểm thử cấu hình thí nghiệm RQ3 (configs/arena/rq3.yaml) và đường cong học tập."""

from __future__ import annotations

from pathlib import Path

from analysis.rq3_learning_curve import (
    build_rq3_markdown,
    load_train_log,
    plot_rq3_learning_curve,
    summarize_arena_rq3,
)
from arena.openings import load_openings
from arena.tournament import build_schedule, load_config, run_tournament
from tests.test_arena import _cfg
from tests.test_train_rl import _make_level5_checkpoint
from training.train_rl import train_rl

ROOT = Path(__file__).resolve().parent.parent
RQ3_CONFIG = ROOT / "configs" / "arena" / "rq3.yaml"


def test_rq3_config_structure_and_schedule():
    cfg = load_config(RQ3_CONFIG)
    agents = cfg["agents"]
    assert agents["sl"]["type"] == "sl"
    assert agents["sl"]["params"]["checkpoint"] == "checkpoints/level5/best.pt"
    for label in ("rl_r005", "rl_r010", "rl_r015", "rl_r020", "rl_best"):
        assert agents[label]["type"] == "rl"
        assert [label, "sl"] in cfg["pairs"]

    tasks = build_schedule(cfg, load_openings(ROOT / cfg["openings_file"]))
    assert len(tasks) == len(cfg["pairs"]) * 200 * len(cfg["seeds"])


def test_rq3_end_to_end_tournament_and_learning_curve(tmp_path):
    sl_ckpt = _make_level5_checkpoint(tmp_path)
    rl_dir = tmp_path / "level6"
    train_rl(
        config={
            "init_checkpoint": str(sl_ckpt),
            "out_dir": str(rl_dir),
            "episodes": 8,
            "episodes_per_round": 4,
            "rollout_batch_games": 4,
            "minibatch_size": 8,
            "eval_games": 2,
            "opening_min_moves": 2,
            "opening_max_moves": 2,
            "device": "cpu",
            "seed": 0,
        },
        out_dir=rl_dir,
    )

    arena_cfg = _cfg(
        tmp_path,
        {
            "sl": {"type": "sl", "params": {"checkpoint": str(sl_ckpt), "device": "cpu"}},
            "rl_r001": {"type": "rl", "params": {"checkpoint": str(rl_dir / "rounds" / "round_001.pt"), "device": "cpu"}},
            "rl_best": {"type": "rl", "params": {"checkpoint": str(rl_dir / "best.pt"), "device": "cpu"}},
        },
        seeds=(0,),
        num_openings=1,
        mode="B",
    )
    arena_cfg["pairs"] = [["rl_r001", "sl"], ["rl_best", "sl"]]
    arena_log = run_tournament(arena_cfg, log_dir=tmp_path / "logs", verbose=False)

    train_records = load_train_log(rl_dir / "train_log.jsonl")
    arena_rows = summarize_arena_rq3(arena_log, baseline_label="sl")
    assert len(train_records) == 3
    assert [r["label"] for r in arena_rows] == ["rl_r001", "rl_best"]

    md = build_rq3_markdown(train_records, arena_rows)
    assert "RQ3" in md
    assert "rl_r001" in md
    assert "Wilson 95%" in md

    fig_path = plot_rq3_learning_curve(
        tmp_path / "figures" / "rq3.png",
        train_records=train_records,
        arena_rows=arena_rows,
    )
    assert fig_path.exists() and fig_path.stat().st_size > 1000
