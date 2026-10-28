"""Pytest tests for Level 2 HeuristicAgent (Issue #3).

Covers all requirements from Issue #3:
1. Thắng ngay (rule: "thắng ngay")
2. Chặn thắng (rule: "chặn thắng")
3. Bốn mở (rule: "bốn mở")
4. Chặn bốn / ba mở (rule: "chặn bốn/ba mở")
5. Đòn kép (rule: "đòn kép")
6. Điểm mẫu hình (rule: "điểm mẫu hình")
7. Log last_stats cập nhật đúng, không ghi file
8. HeuristicAgent đấu thắng áp đảo RandomAgent
"""

import numpy as np
import pytest

from agents.heuristic_agent import (
    RULE_BLOCK_FOUR_OR_THREE,
    RULE_BLOCK_WIN,
    RULE_FORK,
    RULE_OPEN_FOUR,
    RULE_PATTERN_SCORE,
    RULE_WIN,
    HeuristicAgent,
)
from agents.random_agent import RandomAgent
from game.constants import BLACK, WHITE
from game.game import CaroGame


class TestHeuristicRules:
    """Kiểm tra từng luật trong chuỗi ưu tiên của HeuristicAgent."""

    def test_rule_1_immediate_win(self):
        """Ưu tiên 1: Thắng ngay."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        # X có 4 quân: (5, 3), (5, 4), (5, 5), (5, 6)
        # O: (0, 0), (0, 1), (0, 2)
        moves = [
            (5, 3), (0, 0),
            (5, 4), (0, 1),
            (5, 5), (0, 2),
            (5, 6), (0, 3),
        ]
        for m in moves:
            game.step(m)

        # Lượt của X: X phải đánh (5, 7) hoặc (5, 2) để thắng ngay
        action = agent.select_action(game)
        r, c = game.action_to_coord(action)
        assert (r, c) in [(5, 7), (5, 2)]
        assert agent.last_stats == {"rule": RULE_WIN}

    def test_rule_2_block_win(self):
        """Ưu tiên 2: Chặn thắng."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        # X có 4 quân: (5, 3), (5, 4), (5, 5), (5, 6)
        # O có 3 quân không dọa thắng
        # Lượt của O: O phải chặn (5, 7) hoặc (5, 2)
        moves = [
            (5, 3), (0, 0),
            (5, 4), (0, 1),
            (5, 5), (0, 2),
            (5, 6),
        ]
        for m in moves:
            game.step(m)

        action = agent.select_action(game)
        r, c = game.action_to_coord(action)
        assert (r, c) in [(5, 7), (5, 2)]
        assert agent.last_stats == {"rule": RULE_BLOCK_WIN}

    def test_rule_3_open_four(self):
        """Ưu tiên 3: Tạo bốn mở."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        # X có ba mở: (5, 4), (5, 5), (5, 6)
        # O: (0, 0), (0, 1)
        moves = [
            (5, 4), (0, 0),
            (5, 5), (0, 1),
            (5, 6), (1, 1),
        ]
        for m in moves:
            game.step(m)

        # Lượt của X: X đánh (5, 3) hoặc (5, 7) để tạo bốn mở
        action = agent.select_action(game)
        r, c = game.action_to_coord(action)
        assert (r, c) in [(5, 3), (5, 7)]
        assert agent.last_stats == {"rule": RULE_OPEN_FOUR}

    def test_rule_4_block_four_or_three(self):
        """Ưu tiên 4: Chặn bốn / ba mở của đối thủ."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        # X có ba mở: (5, 4), (5, 5), (5, 6)
        # Lượt của O: O phải chặn (5, 3) hoặc (5, 7) để ngăn X tạo bốn mở
        moves = [
            (5, 4), (0, 0),
            (5, 5), (0, 1),
            (5, 6),
        ]
        for m in moves:
            game.step(m)

        action = agent.select_action(game)
        r, c = game.action_to_coord(action)
        assert (r, c) in [(5, 3), (5, 7)]
        assert agent.last_stats == {"rule": RULE_BLOCK_FOUR_OR_THREE}

    def test_rule_5_fork(self):
        """Ưu tiên 5: Đòn kép (3-3 hoặc 4-3)."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        # X có thế cờ chuẩn bị tạo đòn kép tại (5, 5)
        # Ngang: (5, 4), (5, 6)
        # Dọc: (4, 5), (6, 5)
        # O: các ô rời rạc không tạo chuỗi dọa thắng
        moves = [
            (5, 4), (0, 0),
            (5, 6), (0, 5),
            (4, 5), (0, 10),
            (6, 5), (1, 10),
        ]
        for m in moves:
            game.step(m)

        # Lượt của X: X đánh (5, 5) tạo đòn kép 3-3
        action = agent.select_action(game)
        r, c = game.action_to_coord(action)
        assert (r, c) == (5, 5)
        assert agent.last_stats == {"rule": RULE_FORK}

    def test_rule_6_pattern_score_first_move(self):
        """Nước đầu tiên của ván cờ đánh vào trung tâm."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        action = agent.select_action(game)
        assert action == game.coord_to_action(5, 5)
        assert agent.last_stats == {"rule": RULE_PATTERN_SCORE}

    def test_state_not_mutated(self):
        """Tác tử không được làm thay đổi trạng thái state truyền vào."""
        agent = HeuristicAgent(seed=42)
        game = CaroGame(size=11)
        game.step((5, 5))
        game.step((5, 6))

        board_before = game.board.copy()
        current_player_before = game.current_player
        move_count_before = game.move_count

        _ = agent.select_action(game)

        assert np.array_equal(game.board, board_before)
        assert game.current_player == current_player_before
        assert game.move_count == move_count_before

    def test_reproducibility_with_seed(self):
        """Cùng seed phải sinh ra cùng kết quả."""
        agent1 = HeuristicAgent(seed=99)
        agent2 = HeuristicAgent(seed=99)
        game = CaroGame(size=11)
        game.step((5, 5))

        assert agent1.select_action(game) == agent2.select_action(game)


class TestHeuristicVsRandom:
    """HeuristicAgent đối đầu RandomAgent."""

    def test_heuristic_beats_random(self):
        """HeuristicAgent phải thắng áp đảo RandomAgent (100% ván)."""
        heuristic = HeuristicAgent(seed=42)
        random_agent = RandomAgent(seed=100)

        # Chơi 4 ván: 2 ván Heuristic đi trước (X), 2 ván đi sau (O)
        results = []
        for i in range(4):
            game = CaroGame(size=11)
            heuristic_is_black = (i % 2 == 0)

            while not game.is_terminal():
                if (game.current_player == BLACK and heuristic_is_black) or (
                    game.current_player == WHITE and not heuristic_is_black
                ):
                    act = heuristic.select_action(game)
                else:
                    act = random_agent.select_action(game)
                game.step(act)

            expected_winner = BLACK if heuristic_is_black else WHITE
            assert game.winner == expected_winner, (
                f"Ván {i+1}: HeuristicAgent thua RandomAgent (winner={game.winner})"
            )
            results.append(game.winner)

        assert len(results) == 4
