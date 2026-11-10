"""Kiểm thử giao diện pygame ở chế độ không cửa sổ (SDL dummy)."""

import os
import time

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from game.constants import BLACK, WHITE  # noqa: E402
from ui.app import BOARD_ORIGIN, CELL, MODE_AI_VS_AI, WINDOW_SIZE, App, wrap_text  # noqa: E402


@pytest.fixture()
def app():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    application = App(screen)
    yield application
    application.worker.cancel()


def click(app: App, pos):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))


def cell_center(row: int, col: int):
    return BOARD_ORIGIN[0] + col * CELL + CELL // 2, BOARD_ORIGIN[1] + row * CELL + CELL // 2


def run_until(app: App, condition, timeout: float = 15.0):
    end = time.time() + timeout
    while time.time() < end:
        app.update(0.05)
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("Hết thời gian chờ")


def test_menu_builds_and_draws(app, monkeypatch, tmp_path):
    import ui.levels as levels
    from agents.random_agent import RandomAgent

    monkeypatch.setattr(levels, "CARO_ROOT", tmp_path)
    monkeypatch.setattr(levels, "AGENT_TYPES", {**levels.AGENT_TYPES, "puct": RandomAgent})
    app._build_menu()
    app.draw()
    assert app.state == "menu"
    unavailable = [b for b in app.buttons if "(chưa có)" in b.text]
    assert len(unavailable) == 4 and not any(b.enabled for b in unavailable)


def test_start_game_handles_missing_checkpoint_without_crashing(app, monkeypatch, tmp_path):
    import ui.levels as levels
    from agents.random_agent import RandomAgent

    monkeypatch.setattr(levels, "CARO_ROOT", tmp_path)
    monkeypatch.setattr(levels, "AGENT_TYPES", {**levels.AGENT_TYPES, "puct": RandomAgent})
    app.cfg.ai_level = 7  # PUCT đã đăng ký nhưng không có file checkpoint
    app.start_game()
    assert app.state == "menu"
    assert app.session is None
    assert "Không tạo được tác tử" in app.message
    app.draw()


def test_human_vs_ai_flow(app):
    app.cfg.ai_level = 1
    app.start_game()
    assert app.session.is_human_turn
    click(app, cell_center(5, 5))
    assert app.session.moves[0].action == 60 and app.session.moves[0].by_human
    run_until(app, lambda: len(app.session.moves) == 2)
    assert not app.session.moves[1].by_human
    # bấm vào ô đã có quân hoặc ngoài bàn cờ thì bỏ qua
    click(app, cell_center(5, 5))
    click(app, (5, 5))
    assert len(app.session.moves) == 2
    app.draw()


def test_click_ignored_on_ai_turn(app):
    app.cfg.human_side = WHITE
    app.cfg.ai_level = 1
    app.start_game()
    assert app.session.is_ai_turn
    click(app, cell_center(0, 0))
    assert app.session.moves == []


def test_undo_returns_to_human_turn(app):
    app.cfg.ai_level = 1
    app.start_game()
    click(app, cell_center(5, 5))
    run_until(app, lambda: len(app.session.moves) == 2)
    app.undo()
    assert app.session.moves == [] and app.session.is_human_turn


def test_undo_while_ai_thinking_recreates_agent(app):
    app.cfg.ai_level = 1
    app.start_game()
    old_agent = app.session.participants[WHITE].agent
    click(app, cell_center(5, 5))
    app.update(0.05)  # kích hoạt luồng AI suy nghĩ
    app.undo()
    assert app.session.moves == [] and app.session.is_human_turn
    assert app.pending_token is None
    assert app.session.participants[WHITE].agent is not old_agent


def test_ai_vs_ai_runs_pause_and_step(app):
    app.cfg.mode = MODE_AI_VS_AI
    app.cfg.level_x = 1
    app.cfg.level_o = 2
    app.cfg.budget = 0.1
    app.start_game()
    run_until(app, lambda: len(app.session.moves) >= 3)
    app.toggle_pause()
    time.sleep(0.5)
    run_until(app, lambda: app.pending_token is None)
    count = len(app.session.moves)
    for _ in range(10):
        app.update(0.05)
        time.sleep(0.02)
    assert len(app.session.moves) == count  # đang tạm dừng
    app.do_step_once()
    run_until(app, lambda: len(app.session.moves) == count + 1)
    app.draw()


def test_replay_mode_navigation(app):
    app.cfg.ai_level = 1
    app.start_game()
    click(app, cell_center(5, 5))
    run_until(app, lambda: len(app.session.moves) == 2)
    app.toggle_replay()
    assert app.replay is not None and app.replay.index == 2
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    assert app.replay.index == 1
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME))
    assert app.replay.index == 0
    app.draw()
    # trong lúc xem lại, bấm bàn cờ không đi nước mới
    click(app, cell_center(0, 0))
    assert len(app.session.moves) == 2
    app.toggle_replay()
    assert app.replay is None


def test_new_game_and_back_to_menu(app):
    app.cfg.ai_level = 1
    app.start_game()
    click(app, cell_center(5, 5))
    app.new_game()
    assert app.session.moves == [] and app.state == "game"
    app.back_to_menu()
    assert app.state == "menu" and app.session is None


def test_agent_failure_falls_back(app):
    class Broken:
        last_stats: dict = {}

        def select_action(self, state, budget=None):
            raise RuntimeError("hỏng")

    app.cfg.human_side = WHITE
    app.cfg.ai_level = 1
    app.start_game()
    app.session.participants[BLACK].agent = Broken()
    run_until(app, lambda: len(app.session.moves) == 1)
    assert "lỗi" in app.session.moves[0].stats
    assert app.message


def test_cell_at_bounds(app):
    assert app.cell_at(cell_center(0, 0)) == 0
    assert app.cell_at(cell_center(10, 10)) == 120
    assert app.cell_at((0, 0)) is None
    assert app.cell_at((BOARD_ORIGIN[0] + 11 * CELL + 1, BOARD_ORIGIN[1])) is None


def test_wrap_text(app):
    lines = wrap_text(app.font_small, "một hai ba bốn năm sáu bảy tám chín mười", 100)
    assert len(lines) > 1 and all(app.font_small.size(line)[0] <= 100 or " " not in line for line in lines)

