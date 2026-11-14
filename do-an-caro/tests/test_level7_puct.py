"""Kiểm thử cấp 7 (issue #13): PUCT dùng mạng cấp 5 và cấu hình thí nghiệm RQ2."""

from __future__ import annotations

from pathlib import Path

from agents.puct_agent import PUCTAgent
from arena.agent_factory import AgentSpec, build_agent
from arena.logger import load_records
from arena.openings import load_openings
from arena.tournament import build_schedule, load_config, run_tournament
from game.game import CaroGame
from tests.test_arena import _cfg
from tests.test_train_supervised_policy_value_network import _create_synthetic_dataset
from training.train_supervised_policy_value_network import train_sl

ROOT = Path(__file__).resolve().parent.parent
RQ2_CONFIG = ROOT / "configs" / "arena" / "rq2.yaml"


def _train_level5(tmp_path: Path) -> Path:
    """Chạy đúng script train cấp 5 trên dữ liệu giả để có checkpoint thật (kèm optimizer, epoch...)."""
    data = _create_synthetic_dataset(tmp_path / "data.npz", num_samples=8)
    out_dir = tmp_path / "level5"
    train_sl(config={
        "data_path": str(data), "out_dir": str(out_dir), "blocks": 1, "channels": 8,
        "board_size": 11, "epochs": 1, "batch_size": 4, "augment": False,
        "device": "cpu", "seed": 0, "num_workers": 0,
    }, out_dir=out_dir)
    return out_dir / "best.pt"


def test_puct_loads_level5_checkpoint_from_training(tmp_path):
    ckpt = _train_level5(tmp_path)
    agent = build_agent(AgentSpec("cap7", "puct", {"checkpoint": str(ckpt), "simulations": 16}))
    assert isinstance(agent, PUCTAgent)
    game = CaroGame()
    for a in (60, 61, 49):
        game.step(a)
    action = agent.select_action(game)
    assert action in game.legal_actions()
    assert agent.last_stats["sims"] == 16
    assert -1.0 <= agent.last_stats["value"] <= 1.0


def test_rq2_config_compares_same_network_at_same_simulations():
    cfg = load_config(RQ2_CONFIG)
    agents = cfg["agents"]
    level5 = agents["policy"]["params"]["checkpoint"]
    assert agents["policy"]["type"] == "sl"
    for n in (400, 800, 1600):
        puct, mcts = agents[f"puct_{n}"], agents[f"mcts_{n}"]
        assert puct["type"] == "puct" and mcts["type"] == "mcts"
        assert puct["params"]["checkpoint"] == level5          # cùng mạng với policy-only
        assert puct["params"]["simulations"] == mcts["params"]["simulations"] == n
        for pair in ([f"puct_{n}", f"mcts_{n}"], [f"puct_{n}", "policy"], [f"mcts_{n}", "policy"]):
            assert pair in cfg["pairs"]

    tasks = build_schedule(cfg, load_openings(ROOT / cfg["openings_file"]))
    # Chỉ cặp có MCTS (ngẫu nhiên) chạy đủ các seed; cặp PUCT–policy tất định chạy seed đầu
    expected = sum(200 * (len(cfg["seeds"]) if "mcts" in " ".join(pair) else 1) for pair in cfg["pairs"])
    assert len(tasks) == expected == 4200
    assert len({t.game_id for t in tasks}) == len(tasks)


def test_rq2_mini_tournament_runs_without_errors(tmp_path):
    ckpt = str(_train_level5(tmp_path))
    cfg = _cfg(tmp_path, {
        "policy": {"type": "sl", "params": {"checkpoint": ckpt, "device": "cpu"}},
        "puct": {"type": "puct", "params": {"checkpoint": ckpt, "simulations": 8}},
        "mcts": {"type": "mcts", "params": {"simulations": 8}},
    }, seeds=(0,), num_openings=1, mode="B")
    cfg["pairs"] = [["puct", "mcts"], ["puct", "policy"], ["mcts", "policy"]]
    log = run_tournament(cfg, log_dir=tmp_path / "logs", verbose=False)
    records = load_records(log)
    assert len(records) == 6
    assert all(r.get("error") is None for r in records)
    puct_moves = []
    for r in records:
        for side, player in (("agent_x", 1), ("agent_o", -1)):
            if r[side]["label"] == "puct":
                puct_moves += [m for m in r["moves"] if m["player"] == player]
    assert puct_moves and all(m["stats"]["sims"] == 8 for m in puct_moves)
