"""Kiểm thử tác tử Cấp 6 (RLAgent)."""

from __future__ import annotations

import torch

from agents import PUCTAgent, RLAgent, SLAgent
from agents.network import PolicyValueNet, save_checkpoint
from arena.agent_factory import AGENT_TYPES, AgentSpec, build_agent
from game.game import CaroGame


def test_rl_agent_registration_and_inheritance(tmp_path):
    assert issubclass(RLAgent, SLAgent)
    assert AGENT_TYPES["rl"] is RLAgent

    torch.manual_seed(0)
    net = PolicyValueNet(blocks=1, channels=8, size=11)
    ckpt = save_checkpoint(tmp_path / "best.pt", net, epoch=1)

    agent = build_agent(AgentSpec("cap6", "rl", {"checkpoint": str(ckpt), "device": "cpu"}))
    assert isinstance(agent, RLAgent)
    assert agent.name == "rl"

    game = CaroGame()
    game.step(60)
    action = agent.select_action(game)
    assert action in game.legal_actions()
    assert agent.last_stats["action"] == action
    assert -1.0 <= agent.last_stats["value"] <= 1.0


def test_puct_can_also_use_level6_checkpoint(tmp_path):
    torch.manual_seed(1)
    net = PolicyValueNet(blocks=1, channels=8, size=11)
    ckpt = save_checkpoint(tmp_path / "level6_best.pt", net, epoch=5)

    puct = build_agent(AgentSpec("puct_rl", "puct", {"checkpoint": str(ckpt), "simulations": 8, "device": "cpu"}))
    assert isinstance(puct, PUCTAgent)
    game = CaroGame()
    game.step(60)
    assert puct.select_action(game) in game.legal_actions()
