"""Kiểm thử logic giao diện (không cần cửa sổ pygame)."""

import time

import pytest

from agents.random_agent import RandomAgent
from game.constants import BLACK, WHITE
from game.game import CaroGame
from ui.ai_worker import AIWorker
from ui.levels import BUDGET_CHOICES, LEVELS, create_agent, get_level, is_available
from ui.session import GameSession, Participant, Replay, find_win_line
from ui.stats_format import format_stats


def human() -> Participant:
    return Participant()


def bot(level: int = 1, seed: int = 0) -> Participant:
    return Participant(level=level, agent=RandomAgent(seed=seed))


# ---- levels ----
def test_levels_cover_1_to_8():
    assert [info.level for info in LEVELS] == list(range(1, 9))
    assert BUDGET_CHOICES == (0.1, 0.5, 1.0)


def test_levels_1_to_4_available_and_create_agents():
    for level in (1, 2, 3, 4):
        assert is_available(level)
        assert create_agent(level, seed=1) is not None


def test_unregistered_level_unavailable(monkeypatch):
    import ui.levels as levels

    monkeypatch.setattr(levels, "AGENT_TYPES", {"random": RandomAgent})
    assert not levels.is_available(5)
    with pytest.raises(ValueError):
        levels.create_agent(5)


def test_level_requires_checkpoint_and_turns_on_when_present(monkeypatch, tmp_path):
    import ui.levels as levels

    class DummyCheckpointAgent(RandomAgent):
        def __init__(self, checkpoint: str, seed: int | None = None):
            super().__init__(seed=seed)
            self.checkpoint = checkpoint

    monkeypatch.setattr(levels, "CARO_ROOT", tmp_path)
    monkeypatch.setattr(
        levels,
        "AGENT_TYPES",
        {**levels.AGENT_TYPES, get_level(7).agent_type: DummyCheckpointAgent},
    )
    # Dù "puct" đã có trong AGENT_TYPES, thiếu file checkpoint thì vẫn báo chưa sẵn sàng
    assert not levels.is_available(7)
    with pytest.raises(ValueError, match="checkpoint"):
        levels.create_agent(7)

    # Khi file checkpoint tồn tại thì cấp 7 tự bật và tạo được tác tử
    ckpt_rel = get_level(7).params["checkpoint"]
    ckpt_file = tmp_path / ckpt_rel
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt_file.write_bytes(b"dummy")

    assert levels.is_available(7)
    agent = levels.create_agent(7, seed=42)
    assert isinstance(agent, DummyCheckpointAgent)
    assert agent.checkpoint == str(ckpt_file)


def test_get_level_invalid():
    with pytest.raises(ValueError):
        get_level(9)


# ---- session ----
def test_play_records_moves_and_alternates():
    s = GameSession(human(), human())
    r1 = s.play(60)
    r2 = s.play(61, stats={"nodes": 3}, elapsed=0.1)
    assert (r1.player, r2.player) == (BLACK, WHITE)
    assert r1.by_human and r2.stats == {"nodes": 3}
    assert len(s.moves) == 2


def test_play_rejects_illegal_move():
    s = GameSession(human(), human())
    s.play(0)
    with pytest.raises(ValueError):
        s.play(0)


def test_turn_flags():
    s = GameSession(human(), bot())
    assert s.is_human_turn and not s.is_ai_turn
    s.play(0)
    assert s.is_ai_turn and not s.is_human_turn


def test_undo_human_vs_ai_removes_pair():
    s = GameSession(human(), bot())
    s.play(0)
    s.play(1)  # AI đi
    assert s.undo() == 2
    assert s.game.move_count == 0 and s.is_human_turn


def test_undo_when_game_over_after_human_win():
    s = GameSession(human(), bot())
    for human_move, ai_move in [(0, 11), (1, 12), (2, 13), (3, 14)]:
        s.play(human_move)
        s.play(ai_move)
    s.play(4)
    assert s.is_over and s.result_text() == "X thắng"
    assert s.undo() == 1
    assert not s.is_over and s.is_human_turn


def test_undo_when_ai_moves_first_stops_at_start():
    s = GameSession(bot(), human())
    s.play(60)  # AI đi trước
    s.play(61)  # người
    assert s.undo() == 1  # chỉ bỏ nước của người, về lượt người
    assert s.game.move_count == 1


def test_no_undo_for_ai_vs_ai():
    s = GameSession(bot(), bot())
    s.play(0)
    assert not s.can_undo()
    assert s.undo() == 0


def test_reset():
    s = GameSession(human(), human())
    s.play(0)
    s.reset()
    assert s.moves == [] and s.game.move_count == 0


def test_result_text_draw_and_empty():
    s = GameSession(human(), human())
    assert s.result_text() == ""


def test_find_win_line():
    g = CaroGame()
    for a, b in [(0, 11), (1, 12), (2, 13), (3, 14)]:
        g.step(a)
        g.step(b)
    g.step(4)
    assert find_win_line(g) == [0, 1, 2, 3, 4]
    assert find_win_line(CaroGame()) == []


# ---- replay ----
def make_moves():
    s = GameSession(human(), human())
    for a in (60, 61, 71, 72, 82):
        s.play(a, stats={"i": a})
    return s


def test_replay_navigation():
    s = make_moves()
    r = Replay(s.game.size, s.game.rule, s.moves)
    assert r.index == 5 and r.total == 5
    r.first()
    assert r.game.move_count == 0 and r.record is None
    r.next()
    assert r.game.move_count == 1 and r.record.action == 60
    r.last()
    assert r.index == 5
    r.prev()
    assert r.index == 4 and r.game.board[5, 5] == BLACK  # ô 60 = hàng 5, cột 5
    r.goto(99)
    assert r.index == 5
    r.goto(-3)
    assert r.index == 0


def test_replay_game_matches_original():
    s = make_moves()
    r = Replay(s.game.size, s.game.rule, s.moves)
    assert (r.game.board == s.game.board).all()


def test_replay_autoplay_advances_and_stops():
    s = make_moves()
    r = Replay(s.game.size, s.game.rule, s.moves)
    r.first()
    r.toggle_autoplay()
    assert r.autoplay
    r.tick(0.6, interval=0.6)
    assert r.index == 1
    r.tick(10, interval=0.6)
    assert r.index == 5 and not r.autoplay


def test_replay_autoplay_restarts_from_end():
    s = make_moves()
    r = Replay(s.game.size, s.game.rule, s.moves)
    r.toggle_autoplay()
    assert r.index == 0 and r.autoplay


# ---- stats ----
def test_format_stats_alphabeta():
    lines = format_stats({"nodes": 28420, "depth_reached": 4, "score": 12.5, "time_used": 0.323})
    assert "Số nút duyệt: 28.420" in lines
    assert "Độ sâu đạt được: 4" in lines
    assert "Thời gian suy nghĩ: 0.323 giây" in lines


def test_format_stats_mcts_and_none():
    lines = format_stats(
        {"sims": 1600, "win_rate": 0.5, "best_visits": None, "value": 0.25, "prior_best": 0.125}
    )
    assert "Tỷ lệ thắng ước tính: 50.0%" in lines
    assert "Lượt thăm nước tốt nhất: không có" in lines
    assert "Giá trị dự đoán: 0.25" in lines
    assert "Xác suất prior nước chọn: 12.5%" in lines


def test_format_stats_empty_uses_elapsed():
    assert format_stats({}, elapsed=0.0123) == ["Thời gian suy nghĩ: 0.012 giây"]
    assert format_stats({}) == []


def test_format_stats_unknown_key_kept():
    assert format_stats({"foo": 1}) == ["foo: 1"]


# ---- worker ----
def wait(worker: AIWorker, timeout: float = 5.0):
    end = time.time() + timeout
    while time.time() < end:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("AIWorker không trả kết quả")


def test_worker_returns_legal_move_without_touching_state():
    g = CaroGame()
    before = g.board.copy()
    w = AIWorker()
    token = w.start(RandomAgent(seed=3), g, budget=0.1)
    res = wait(w)
    assert res.token == token and res.error is None
    assert res.action in g.legal_actions()
    assert (g.board == before).all()
    assert w.poll() is None


def test_worker_cancel_drops_result():
    class Slow(RandomAgent):
        def select_action(self, state, budget=None):
            time.sleep(0.2)
            return super().select_action(state, budget)

    w = AIWorker()
    w.start(Slow(seed=1), CaroGame(), None)
    w.cancel()
    time.sleep(0.4)
    assert w.poll() is None and not w.busy


def test_worker_reports_agent_error():
    class Broken(RandomAgent):
        def select_action(self, state, budget=None):
            raise RuntimeError("hỏng")

    w = AIWorker()
    w.start(Broken(seed=1), CaroGame(), None)
    res = wait(w)
    assert res.action is None and "hỏng" in res.error


def test_worker_waits_for_cancelled_thread_before_starting_next():
    active = 0
    max_concurrent = 0

    class SlowTracker(RandomAgent):
        def select_action(self, state, budget=None):
            nonlocal active, max_concurrent
            active += 1
            max_concurrent = max(max_concurrent, active)
            try:
                time.sleep(0.15)
                return super().select_action(state, budget)
            finally:
                active -= 1

    agent = SlowTracker(seed=1)
    w = AIWorker()
    w.start(agent, CaroGame(), None)
    time.sleep(0.03)
    w.cancel()
    # Bắt đầu lượt mới ngay khi luồng cũ chưa kịp kết thúc
    token2 = w.start(agent, CaroGame(), None)
    res = wait(w)
    assert res.token == token2
    assert max_concurrent == 1
