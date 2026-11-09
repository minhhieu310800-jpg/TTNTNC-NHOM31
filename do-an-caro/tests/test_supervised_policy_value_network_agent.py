"""Kiểm thử tác tử Cấp 5 (SLAgent / PolicyAgent)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import (
    PolicyAgent,
    SLAgent,
    SupervisedPolicyValueNetworkAgent,
)
from agents.network import NetEvaluator, PolicyValueNet, save_checkpoint
from arena.agent_factory import AgentSpec, build_agent
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

    # PolicyAgent và SupervisedPolicyValueNetworkAgent là alias của SLAgent
    assert PolicyAgent is SLAgent
    assert SupervisedPolicyValueNetworkAgent is SLAgent


def test_sl_agent_select_action_returns_legal_move():
    agent = SLAgent(evaluator=_small_evaluator())
    game = CaroGame()
    legals = set(game.legal_actions())

    action = agent.select_action(game)
    assert action in legals
    assert 0 <= action < 121

    # Kiểm tra thống kê nước đi
    stats = agent.last_stats
    assert "value" in stats
    assert "policy_prob" in stats
    assert "action" in stats
    assert stats["action"] == action
    assert -1.0 <= stats["value"] <= 1.0
    assert 0.0 <= stats["policy_prob"] <= 1.0


def test_sl_agent_plays_multiple_turns_deterministically():
    agent = SLAgent(evaluator=_small_evaluator(), temperature=0.0)
    game = CaroGame()

    # Nước đầu tiên
    a1 = agent.select_action(game)
    a1_again = agent.select_action(game)
    assert a1 == a1_again  # Argmax luôn tất định

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

    # Đăng ký với tên đầy đủ
    spec_full = AgentSpec.from_config("spvn_agent", {
        "type": "supervised_policy_value_network",
        "params": {
            "checkpoint": str(ckpt_path),
            "device": "cpu",
        }
    })
    agent_full = build_agent(spec_full)
    assert isinstance(agent_full, SupervisedPolicyValueNetworkAgent)

    game = CaroGame()
    action = agent.select_action(game)
    assert action in game.legal_actions()


def test_sl_agent_temperature_sampling():
    agent = SLAgent(evaluator=_small_evaluator(), temperature=1.0, seed=123)
    game = CaroGame()
    action = agent.select_action(game)
    assert action in game.legal_actions()

