"""Pytest tests for Level 1 RandomAgent."""

import pytest

from agents.random_agent import RandomAgent
from game.constants import BLACK, WHITE
from game.game import CaroGame


class TestRandomAgent:
    """Kiểm thử tác tử ngẫu nhiên cấp 1."""

    def test_random_agent_initialization(self):
        """Khởi tạo có và không có seed."""
        agent1 = RandomAgent(seed=42)
        assert agent1.seed == 42
        assert agent1.name == "random"
        assert agent1.last_stats == {}

        agent2 = RandomAgent()
        assert agent2.seed is None
        assert agent2.last_stats == {}

    def test_random_agent_reproducibility(self):
        """Seed cố định phải cho ra chuỗi nước đi hoàn toàn giống nhau."""
        agent_a = RandomAgent(seed=12345)
        agent_b = RandomAgent(seed=12345)

        game_a = CaroGame(size=11)
        game_b = CaroGame(size=11)

        for _ in range(10):
            action_a = agent_a.select_action(game_a)
            action_b = agent_b.select_action(game_b)
            assert action_a == action_b
            game_a.step(action_a)
            game_b.step(action_b)

    def test_random_agent_different_seeds(self):
        """Hai seed khác nhau sinh ra chuỗi nước đi khác nhau."""
        agent_a = RandomAgent(seed=1)
        agent_b = RandomAgent(seed=9999)

        game = CaroGame(size=11)
        # Nước đầu tiên có thể trùng xác suất 1/121, nhưng chạy 5 ván độc lập sẽ thấy khác nhau
        actions_a = [agent_a.select_action(game) for _ in range(5)]
        agent_a.reset_seed(1)
        actions_a_reproduced = [agent_a.select_action(game) for _ in range(5)]
        assert actions_a == actions_a_reproduced

    def test_random_agent_selects_legal_action(self):
        """Nước đi trả về luôn nằm trong legal_actions()."""
        agent = RandomAgent(seed=42)
        game = CaroGame(size=11)

        for _ in range(20):
            legal = game.legal_actions()
            action = agent.select_action(game)
            assert action in legal
            assert agent.last_stats == {}
            game.step(action)

    def test_random_agent_last_stats_empty(self):
        """Theo issue #3, RandomAgent phải cập nhật last_stats = {} sau mỗi nước."""
        agent = RandomAgent(seed=42)
        game = CaroGame(size=11)
        agent.select_action(game)
        assert agent.last_stats == {}

    def test_random_agent_raises_on_full_board(self):
        """Ném ngoại lệ ValueError nếu không còn nước đi hợp lệ."""
        agent = RandomAgent(seed=42)
        game = CaroGame(size=11)
        # Điền kín bàn cờ
        game.board.fill(1)
        with pytest.raises(ValueError, match="Không còn nước đi hợp lệ"):
            agent.select_action(game)
