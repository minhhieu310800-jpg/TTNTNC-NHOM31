"""Kiểm thử tác tử Cấp 5 (SLAgent)."""

from __future__ import annotations

import pytest
import torch

from agents import SLAgent
from agents.network import NetEvaluator, PolicyValueNet, save_checkpoint
from agents.puct_agent import cached_evaluator as puct_cached_evaluator
import agents.supervised_policy_value_network_agent as sl_mod
from arena.agent_factory import AGENT_TYPES, AgentSpec, build_agent
from game.game import CaroGame


def _small_evaluator() -> NetEvaluator:
    torch.manual_seed(42)
    net = PolicyValueNet(blocks=1, channels=8, size=11).eval()
    return NetEvaluator(net, device="cpu")


def test_sl_agent_initialization():
    evaluator = _small_evaluator()
    agent = SLAgent(evaluator=evaluator)
    assert agent.name == "sl"
    assert agent.evaluator is evaluator
    assert agent.temperature == 0.0
    assert sl_mod.cached_evaluator is puct_cached_evaluator


def test_sl_agent_select_action_returns_legal_move():
    agent = SLAgent(evaluator=_small_evaluator())
    game = CaroGame()
    legals = set(game.legal_actions())

    action = agent.select_action(game)
    assert action in legals
    assert 0 <= action < 121

    stats = agent.last_stats
    assert "value" in stats
    assert "policy_prob" in stats
    assert "action" in stats
    assert "time_used" in stats
    assert stats["action"] == action
    assert -1.0 <= stats["value"] <= 1.0
    assert 0.0 <= stats["policy_prob"] <= 1.0


def test_sl_agent_updates_last_stats_on_single_legal_move():
    agent = SLAgent(evaluator=_small_evaluator())
    game = CaroGame()

    # Đặt trước 1 nước để tạo last_stats ban đầu
    first_action = agent.select_action(game)
    first_stats = dict(agent.last_stats)

    # Tạo bàn cờ giả lập còn đúng 1 ô trống (không kích hoạt thắng)
    # Xen kẽ X và O sao cho không có hàng 5
    board = game.board
    board.fill(0)
    empty_r, empty_c = 10, 10
    empty_idx = empty_r * 11 + empty_c
    for r in range(11):
        for c in range(11):
            if (r, c) == (empty_r, empty_c):
                continue
            # Mẫu xen kẽ từng cặp 2 quân theo hàng/cột để không tạo chuỗi 5
            board[r, c] = 1 if ((r // 2 + c // 2) % 2 == 0) else -1
    game.move_count = 120
    game.last_move = 0
    game.current_player = 1

    action = agent.select_action(game)
    assert action == empty_idx
    assert agent.last_stats is not None
    assert agent.last_stats["action"] == empty_idx
    assert agent.last_stats != first_stats
    assert agent.last_stats["policy_prob"] == pytest.approx(1.0, abs=1e-6)
    assert -1.0 <= agent.last_stats["value"] <= 1.0


def test_sl_agent_plays_multiple_turns_deterministically():
    agent = SLAgent(evaluator=_small_evaluator(), temperature=0.0)
    game = CaroGame()

    a1 = agent.select_action(game)
    a1_again = agent.select_action(game)
    assert a1 == a1_again

    game.step(a1)
    a2 = agent.select_action(game)
    assert a2 != a1
    assert a2 in game.legal_actions()


def test_sl_agent_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    net = PolicyValueNet(blocks=1, channels=8, size=11)
    ckpt_path = save_checkpoint(tmp_path / "model.pt", net, epoch=1)

    agent = SLAgent(checkpoint=ckpt_path, device="cpu")
    game = CaroGame()
    action = agent.select_action(game)
    assert action in game.legal_actions()


def test_sl_agent_arena_factory_integration(tmp_path):
    assert AGENT_TYPES["sl"] is SLAgent
    assert "policy" not in AGENT_TYPES
    assert "supervised_policy_value_network" not in AGENT_TYPES

    torch.manual_seed(0)
    net = PolicyValueNet(blocks=1, channels=8, size=11)
    ckpt_path = save_checkpoint(tmp_path / "model.pt", net)

    spec = AgentSpec.from_config("sl_agent", {
        "type": "sl",
        "params": {
            "checkpoint": str(ckpt_path),
            "device": "cpu",
        }
    })
    agent = build_agent(spec)
    assert isinstance(agent, SLAgent)

    game = CaroGame()
    action = agent.select_action(game)
    assert action in game.legal_actions()


def test_sl_agent_temperature_sampling():
    agent = SLAgent(evaluator=_small_evaluator(), temperature=1.0, seed=123)
    game = CaroGame()
    action = agent.select_action(game)
    assert action in game.legal_actions()
