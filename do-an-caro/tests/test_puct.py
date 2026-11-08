"""Kiểm thử PUCT (agents/puct_agent.py) và self-play (training/selfplay.py)."""

from __future__ import annotations

import numpy as np
import torch

from agents.network import PolicyValueNet, save_checkpoint
from agents.puct_agent import Node, PUCTAgent, PUCTTree, UniformEvaluator, run_search
from arena.agent_factory import AgentSpec, build_agent
from game.game import CaroGame
from training.selfplay import SelfPlayConfig, play_games, symmetry


def _game(xs, os_, x_to_move=True) -> CaroGame:
    game = CaroGame()
    for i in range(max(len(xs), len(os_))):
        if i < len(xs):
            game.step(xs[i])
        if i < len(os_):
            game.step(os_[i])
    assert (game.current_player == 1) == x_to_move
    return game


def test_finds_immediate_win_with_uniform_network():
    # X có bốn mở ở hàng 5 (cột 3..6), đến lượt X
    game = _game([58, 59, 60, 61], [0, 10, 110, 120])
    agent = PUCTAgent(evaluator=UniformEvaluator(), simulations=300)
    action = agent.select_action(game)
    assert action in (57, 62)
    assert {"sims", "best_visits", "value", "prior_best", "time_used"} <= set(agent.last_stats)
    assert agent.last_stats["value"] > 0.5


def test_search_leaves_game_unchanged_and_policy_is_valid():
    game = _game([60, 48], [61, 72])
    board = game.board.copy()
    tree = PUCTTree(game.current_player)
    sims = run_search(tree, game, UniformEvaluator(), simulations=50)
    assert sims == 50
    assert np.array_equal(game.board, board) and game.move_count == 4
    pi = tree.visit_policy(121)
    assert abs(pi.sum() - 1) < 1e-6
    assert pi[[60, 48, 61, 72]].sum() == 0
    assert tree.root.total == 49  # mô phỏng đầu tiên chỉ mở rộng gốc


def test_advance_keeps_subtree_and_dirichlet_noise_is_valid():
    game = CaroGame()
    # bàn trống + radius chỉ có 1 nước (tâm), nên tắt radius để thử nhiễu
    tree = PUCTTree(1, dirichlet=(0.3, 0.25), rng=np.random.default_rng(0), radius=None)
    run_search(tree, game, UniformEvaluator(), simulations=40)
    assert abs(tree.root.P.sum() - 1) < 1e-6
    assert not np.allclose(tree.root.P, tree.root.P[0])  # nhiễu đã trộn vào prior đều
    best = int(tree.root.actions[np.argmax(tree.root.N)])
    child_visits = int(tree.root.N.max())
    tree.advance(best)
    assert tree.root.total == child_visits - 1
    assert tree.root.to_play == -1


def test_radius_limits_moves_to_neighborhood():
    game = _game([60], [], x_to_move=False)  # X ở tâm (5,5), đến lượt O
    tree = PUCTTree(game.current_player, radius=2)
    run_search(tree, game, UniformEvaluator(), simulations=30)
    rows, cols = np.divmod(tree.root.actions, 11)
    assert len(tree.root.actions) == 24  # ô 5x5 quanh tâm trừ ô đã đánh
    assert np.all(np.abs(rows - 5) <= 2) and np.all(np.abs(cols - 5) <= 2)
    full = PUCTTree(game.current_player, radius=None)
    run_search(full, game, UniformEvaluator(), simulations=2)
    assert len(full.root.actions) == 120


def test_fpu_lowers_q_of_unvisited_moves():
    node = Node(1)
    node.expand(np.array([0.6, 0.3, 0.1]), np.array([True, True, True]))
    node.N[:] = [3, 0, 0]
    node.W[:] = [0.6, 0, 0]          # Q(nước 0) = 0.2, Q(cha) = 0.2
    node.total = 3
    # Không FPU: nước 1 chưa thăm có Q = Q(cha), U lớn hơn nước 0 -> được chọn
    assert node.select(1.0, fpu=0.0) == 1
    # FPU: Q(nước 1) = 0.2 - 1.0 * sqrt(0.6) < 0 -> quay lại nước 0
    assert node.select(1.0, fpu=1.0) == 0


def test_symmetry_moves_planes_and_policy_together():
    game = _game([2 * 11 + 3], [], x_to_move=False)
    planes = game.get_planes()[None]
    pi = np.zeros((1, 121), dtype=np.float32)
    pi[0, 2 * 11 + 3] = 1.0
    for k in range(8):
        p, q = symmetry(planes, pi, k)
        r, c = divmod(int(np.argmax(q[0])), 11)
        assert p[0, 2, r, c] == 1  # nước vừa đi và nhãn policy ở cùng ô
        assert np.array_equal(p[0, 3], planes[0, 3])
    p0, q0 = symmetry(planes, pi, 0)
    assert np.array_equal(p0, planes) and np.array_equal(q0, pi)


def test_selfplay_samples_are_consistent():
    data = play_games(UniformEvaluator(), n_games=2, cfg=SelfPlayConfig(simulations=8, temp_moves=4), seed=1)
    n = len(data["z"])
    assert data["planes"].shape == (n, 4, 11, 11) and data["pi"].shape == (n, 121)
    assert n == int(data["lengths"].sum())
    assert np.allclose(data["pi"].sum(axis=1), 1.0, atol=1e-5)
    occupied = (data["planes"][:, 0] + data["planes"][:, 1]).reshape(n, 121) > 0
    assert np.all(data["pi"][occupied] == 0)
    assert set(np.unique(data["z"])) <= {-1.0, 0.0, 1.0}
    # người đi nước cuối của ván có người thắng nhận z = +1
    first_len = int(data["lengths"][0])
    if data["results"][0] != 0:
        assert data["z"][first_len - 1] == 1.0


def test_puct_agent_from_checkpoint_in_arena(tmp_path):
    torch.manual_seed(0)
    path = save_checkpoint(tmp_path / "az.pt", PolicyValueNet(blocks=1, channels=8))
    agent = build_agent(AgentSpec("az", "puct", {"checkpoint": str(path), "simulations": 10}))
    game = _game([60], [61])
    action = agent.select_action(game)
    assert game.board.ravel()[action] == 0
    assert agent.last_stats["sims"] == 10
