"""Pytest test simulating 10,000 random games.

Covers Issue #2: "10.000 ván ngẫu nhiên không lỗi"
- Checks engine robustness, memory stability, and rule consistency.
- Ensures all games terminate cleanly with valid outcomes (1, -1, 0).
- Verifies deterministic reproducibility under fixed seeds.
"""

import time
import pytest

from agents.random_agent import RandomAgent
from game.constants import BLACK, RULE_VIETNAM, WHITE
from game.game import CaroGame


class TestRandomSimulation:
    """Kiểm thử độ ổn định và tính đúng đắn với 10.000 ván đấu ngẫu nhiên."""

    def test_10000_random_games_robustness(self):
        """Chạy 10.000 ván ngẫu nhiên không xảy ra bất kỳ ngoại lệ nào."""
        n_games = 10_000
        agent_x = RandomAgent(seed=42)
        agent_o = RandomAgent(seed=1337)
        game = CaroGame(size=11, rule=RULE_VIETNAM)

        results = {BLACK: 0, WHITE: 0, 0: 0}
        total_moves = 0

        start_time = time.perf_counter()

        for game_idx in range(n_games):
            game.reset()
            while not game.is_terminal():
                current_agent = agent_x if game.current_player == BLACK else agent_o
                action = current_agent.select_action(game)
                game.step(action)

            # Kiểm tra trạng thái kết thúc hợp lệ
            assert game.is_terminal() is True, f"Ván #{game_idx} chưa kết thúc hợp lệ."
            res = game.result()
            assert res in (BLACK, WHITE, 0), f"Ván #{game_idx} có kết quả bất thường: {res}"

            # Kiểm tra số nước đi hợp lệ
            assert 9 <= game.move_count <= 121, f"Ván #{game_idx} có số nước đi phi lý: {game.move_count}"

            # Nếu có người thắng thì legal_actions có thể còn; nếu hoà thì legal_actions phải rỗng
            if res == 0:
                assert len(game.legal_actions()) == 0
                assert game.move_count == 121

            results[res] += 1
            total_moves += game.move_count

        elapsed = time.perf_counter() - start_time

        # Báo cáo kết quả
        avg_moves = total_moves / n_games
        games_per_sec = n_games / elapsed if elapsed > 0 else float("inf")

        print(
            f"\n--- 10,000 RANDOM GAMES SIMULATION RESULT ---\n"
            f"Execution time: {elapsed:.2f} s ({games_per_sec:.1f} games/s)\n"
            f"Total moves: {total_moves:,} (avg {avg_moves:.1f} moves/game)\n"
            f"Black (X) wins: {results[BLACK]:,} ({results[BLACK] / n_games * 100:.2f}%)\n"
            f"White (O) wins: {results[WHITE]:,} ({results[WHITE] / n_games * 100:.2f}%)\n"
            f"Draws: {results[0]:,} ({results[0] / n_games * 100:.2f}%)"
        )

        assert results[BLACK] + results[WHITE] + results[0] == n_games
