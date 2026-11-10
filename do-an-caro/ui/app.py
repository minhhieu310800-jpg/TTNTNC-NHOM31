"""Giao diện Pygame demo: chơi người-máy, máy-máy, xem lại ván, xem thông số AI cân nhắc.

Chạy (trong thư mục ``do-an-caro``)::

    python -m ui.app

Logic ván cờ nằm ở ``ui/session.py`` và ``ui/ai_worker.py`` (có test); file này chỉ lo
vẽ và nhận sự kiện chuột/bàn phím.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Final

import pygame

from game.constants import BLACK, DEFAULT_BOARD_SIZE, WHITE
from ui.ai_worker import AIWorker
from ui.levels import BUDGET_CHOICES, DEFAULT_BUDGET, LEVELS, create_agent, get_level, is_available
from ui.session import PLAYER_NAMES, GameSession, MoveRecord, Participant, Replay, find_win_line
from ui.stats_format import format_stats

# ---- Cấu hình giao diện ----
WINDOW_SIZE: Final[tuple[int, int]] = (1000, 680)
FPS: Final[int] = 60
CELL: Final[int] = 52
BOARD_ORIGIN: Final[tuple[int, int]] = (50, 70)
PANEL_X: Final[int] = 660
PANEL_W: Final[int] = 320
AI_MOVE_DELAY: Final[float] = 0.4  # nghỉ giữa hai nước ở chế độ máy-máy để dễ theo dõi
REPLAY_INTERVAL: Final[float] = 0.6
FONT_NAMES: Final[str] = "segoeui,tahoma,arial,dejavusans,freesans"

COLOR_BG: Final = (243, 238, 226)
COLOR_BOARD: Final = (250, 246, 232)
COLOR_GRID: Final = (120, 110, 95)
COLOR_TEXT: Final = (40, 40, 40)
COLOR_MUTED: Final = (140, 140, 140)
COLOR_X: Final = (200, 40, 40)
COLOR_O: Final = (30, 80, 190)
COLOR_LAST: Final = (255, 226, 110)
COLOR_WIN: Final = (140, 215, 140)
COLOR_HOVER: Final = (225, 225, 205)
COLOR_BTN: Final = (225, 220, 205)
COLOR_BTN_SELECTED: Final = (90, 130, 200)
COLOR_BTN_DISABLED: Final = (232, 230, 224)
COLOR_PANEL: Final = (255, 255, 250)

MODE_HUMAN_VS_AI: Final[str] = "hvm"
MODE_AI_VS_AI: Final[str] = "mvm"


@dataclass
class Button:
    """Nút bấm chữ nhật đơn giản."""

    rect: pygame.Rect
    text: str
    on_click: Callable[[], None]
    enabled: bool = True
    selected: bool = False

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


@dataclass
class MenuConfig:
    """Lựa chọn ở màn hình menu."""

    mode: str = MODE_HUMAN_VS_AI
    human_side: int = BLACK
    ai_level: int = 3
    level_x: int = 3
    level_o: int = 2
    budget: float = DEFAULT_BUDGET


def _make_font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont(FONT_NAMES, size, bold=bold)


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    """Ngắt dòng theo từ để vừa ``max_width`` pixel."""
    words, lines, current = text.split(" "), [], ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if font.size(trial)[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class App:
    """Ứng dụng: màn hình menu và màn hình chơi/xem lại."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = _make_font(18)
        self.font_small = _make_font(16)
        self.font_big = _make_font(26, bold=True)
        self.font_h = _make_font(21, bold=True)
        self.font_title = _make_font(40, bold=True)
        self.cfg = MenuConfig()
        self.state = "menu"  # "menu" | "game"
        self.buttons: list[Button] = []
        self._title_at: list[tuple[str, int, int]] = []
        self.session: GameSession | None = None
        self.replay: Replay | None = None
        self.worker = AIWorker()
        self.pending_token: int | None = None
        self.paused = False
        self.step_once = False
        self.last_move_clock = 0.0
        self.message = ""
        self.mouse = (0, 0)
        self.running = True
        self._build_menu()

    # ------------------------------------------------------------------ menu
    def _add(self, x: int, y: int, w: int, h: int, text: str, on_click: Callable[[], None],
             enabled: bool = True, selected: bool = False) -> None:
        self.buttons.append(Button(pygame.Rect(x, y, w, h), text, on_click, enabled, selected))

    def _set(self, name: str, value) -> Callable[[], None]:
        def action() -> None:
            setattr(self.cfg, name, value)
            self._build_menu()

        return action

    def _level_column(self, x: int, y: int, attr: str, title: str) -> None:
        """Cột 8 nút chọn cấp AI; cấp chưa có tác tử bị làm mờ."""
        self._title_at.append((title, x, y - 28))
        for i, info in enumerate(LEVELS):
            available = is_available(info.level)
            label = info.title if available else f"{info.title} (chưa có)"
            self._add(x, y + i * 38, 300, 32, label, self._set(attr, info.level),
                      enabled=available, selected=getattr(self.cfg, attr) == info.level)

    def _build_menu(self) -> None:
        self.state = "menu"
        self.buttons = []
        self._title_at = []
        cfg = self.cfg
        self._add(50, 130, 220, 40, "Người – Máy", self._set("mode", MODE_HUMAN_VS_AI),
                  selected=cfg.mode == MODE_HUMAN_VS_AI)
        self._add(280, 130, 220, 40, "Máy – Máy", self._set("mode", MODE_AI_VS_AI),
                  selected=cfg.mode == MODE_AI_VS_AI)
        if cfg.mode == MODE_HUMAN_VS_AI:
            self._title_at.append(("Bạn cầm quân", 50, 195))
            self._add(50, 225, 220, 36, "X (đi trước)", self._set("human_side", BLACK),
                      selected=cfg.human_side == BLACK)
            self._add(280, 225, 220, 36, "O (đi sau)", self._set("human_side", WHITE),
                      selected=cfg.human_side == WHITE)
            self._level_column(50, 330, "ai_level", "Đối thủ AI")
        else:
            self._level_column(50, 230, "level_x", "AI cầm quân X (đi trước)")
            self._level_column(400, 230, "level_o", "AI cầm quân O (đi sau)")
        self._title_at.append(("Thời gian suy nghĩ", 720, 195))
        for i, budget in enumerate(BUDGET_CHOICES):
            self._add(720, 225 + i * 44, 220, 36, f"{budget:.1f} giây".replace(".", ","),
                      self._set("budget", budget), selected=cfg.budget == budget)
        self._add(720, 520, 220, 52, "Bắt đầu", self.start_game, selected=True)

    # ------------------------------------------------------------------ game
    def _participant(self, level: int) -> Participant:
        return Participant(level=level, agent=create_agent(level))

    def _recreate_ai_agents(self) -> None:
        """Khởi tạo lại tác tử AI để tránh dùng chung trạng thái với luồng cũ đã hủy."""
        if self.session is None:
            return
        for participant in self.session.participants.values():
            if not participant.is_human and participant.level is not None:
                try:
                    participant.agent = create_agent(participant.level)
                except Exception:
                    pass

    def start_game(self) -> None:
        """Tạo ván mới theo cấu hình menu và chuyển sang màn hình chơi."""
        cfg = self.cfg
        try:
            if cfg.mode == MODE_HUMAN_VS_AI:
                ai = self._participant(cfg.ai_level)
                black, white = (Participant(), ai) if cfg.human_side == BLACK else (ai, Participant())
            else:
                black, white = self._participant(cfg.level_x), self._participant(cfg.level_o)
        except Exception as exc:
            self.message = f"Không tạo được tác tử: {exc}"
            self._build_menu()
            return
        self.session = GameSession(black, white, budget=cfg.budget, size=DEFAULT_BOARD_SIZE)
        self._enter_game()

    def _enter_game(self) -> None:
        self.worker.cancel()
        self.pending_token = None
        self.replay = None
        self.paused = False
        self.step_once = False
        self.message = ""
        self.last_move_clock = time.time()
        self.state = "game"
        self._build_game()

    def new_game(self) -> None:
        assert self.session is not None
        self.worker.cancel()
        self._recreate_ai_agents()
        self.session.reset()
        self._enter_game()

    def back_to_menu(self) -> None:
        self.worker.cancel()
        self.pending_token = None
        self.session = None
        self.replay = None
        self.message = ""
        self._build_menu()

    def undo(self) -> None:
        assert self.session is not None
        self.worker.cancel()
        self.pending_token = None
        self._recreate_ai_agents()
        self.session.undo()
        self.message = "Đã hoàn tác."
        self._build_game()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._build_game()

    def do_step_once(self) -> None:
        self.step_once = True

    def toggle_replay(self) -> None:
        assert self.session is not None
        if self.replay is None:
            self.replay = Replay(self.session.game.size, self.session.game.rule, self.session.moves)
        else:
            self.replay = None
        self._build_game()

    def _replay_action(self, name: str) -> Callable[[], None]:
        def action() -> None:
            if self.replay is not None:
                getattr(self.replay, name)()
                self._build_game()

        return action

    def _build_game(self) -> None:
        self.buttons = []
        s = self.session
        assert s is not None
        x, y, w = PANEL_X, 400, PANEL_W
        half = (w - 10) // 2
        self._add(x, y, half, 34, "Ván mới", self.new_game)
        self._add(x + half + 10, y, half, 34, "Về menu", self.back_to_menu)
        if s.has_human:
            self._add(x, y + 44, w, 34, "Hoàn tác (lùi về lượt bạn)", self.undo,
                      enabled=s.can_undo() and self.replay is None)
        else:
            self._add(x, y + 44, half, 34, "Tiếp tục" if self.paused else "Tạm dừng", self.toggle_pause,
                      enabled=not s.is_over and self.replay is None)
            self._add(x + half + 10, y + 44, half, 34, "Đi 1 nước", self.do_step_once,
                      enabled=self.paused and not s.is_over and self.replay is None)
        self._add(x, y + 88, w, 34, "Thoát xem lại" if self.replay else "Xem lại ván",
                  self.toggle_replay, enabled=bool(s.moves), selected=self.replay is not None)
        if self.replay is not None:
            bw = (w - 4 * 6) // 5
            labels = [("|<", "first"), ("<", "prev"), (">", "next"), (">|", "last")]
            for i, (text, name) in enumerate(labels):
                self._add(x + i * (bw + 6), y + 132, bw, 34, text, self._replay_action(name))
            self._add(x + 4 * (bw + 6), y + 132, bw, 34, "Dừng" if self.replay.autoplay else "Phát",
                      self._replay_action("toggle_autoplay"))

    # ------------------------------------------------------------------ events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.enabled and button.contains(event.pos):
                    button.on_click()
                    return
            if self.state == "game":
                self._click_board(event.pos)
        elif event.type == pygame.KEYDOWN and self.state == "game" and self.replay is not None:
            keys = {pygame.K_LEFT: "prev", pygame.K_RIGHT: "next", pygame.K_HOME: "first", pygame.K_END: "last"}
            if event.key in keys:
                self._replay_action(keys[event.key])()
            elif event.key == pygame.K_SPACE:
                self._replay_action("toggle_autoplay")()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False

    def cell_at(self, pos: tuple[int, int]) -> int | None:
        """Ô (chỉ số action) tại tọa độ chuột; ``None`` nếu ngoài bàn cờ."""
        col = (pos[0] - BOARD_ORIGIN[0]) // CELL
        row = (pos[1] - BOARD_ORIGIN[1]) // CELL
        if pos[0] < BOARD_ORIGIN[0] or pos[1] < BOARD_ORIGIN[1]:
            return None
        if 0 <= row < DEFAULT_BOARD_SIZE and 0 <= col < DEFAULT_BOARD_SIZE:
            return row * DEFAULT_BOARD_SIZE + col
        return None

    def _click_board(self, pos: tuple[int, int]) -> None:
        s = self.session
        if s is None or self.replay is not None or not s.is_human_turn:
            return
        action = self.cell_at(pos)
        if action is not None and action in s.game.legal_actions():
            s.play(action)
            self.message = ""
            self.last_move_clock = time.time()
            self._build_game()

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        if self.state != "game" or self.session is None:
            return
        if self.replay is not None:
            was_autoplay = self.replay.autoplay
            self.replay.tick(dt, REPLAY_INTERVAL)
            if was_autoplay and not self.replay.autoplay:
                self._build_game()
            return
        s = self.session
        result = self.worker.poll()
        if result is not None and result.token == self.pending_token:
            self.pending_token = None
            self._apply_ai_result(result)
            return
        if not s.is_ai_turn or self.pending_token is not None:
            return
        automatic = s.has_human or (not self.paused) or self.step_once
        if not automatic:
            return
        if not s.has_human and time.time() - self.last_move_clock < AI_MOVE_DELAY:
            return
        self.step_once = False
        agent = s.current.agent
        assert agent is not None
        self.pending_token = self.worker.start(agent, s.game, s.budget)

    def _apply_ai_result(self, result) -> None:
        s = self.session
        assert s is not None
        legal = s.game.legal_actions()
        stats = dict(result.stats)
        action = result.action
        if result.error is not None or action not in legal:
            self.message = f"Tác tử lỗi, đi tạm ô ngẫu nhiên ({result.error or 'nước không hợp lệ'})."
            stats = {"lỗi": result.error or "nước không hợp lệ"}
            action = random.choice(legal)
        s.play(action, stats=stats, elapsed=result.elapsed)
        self.last_move_clock = time.time()
        self._build_game()

    # ------------------------------------------------------------------ draw
    def draw(self) -> None:
        self.screen.fill(COLOR_BG)
        if self.state == "menu":
            self._draw_menu()
        else:
            self._draw_game()
        for button in self.buttons:
            self._draw_button(button)

    def _text(self, text: str, pos: tuple[int, int], font: pygame.font.Font | None = None,
              color: tuple[int, int, int] = COLOR_TEXT) -> int:
        font = font or self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)
        return surface.get_height()

    def _draw_button(self, b: Button) -> None:
        if not b.enabled:
            fill, color = COLOR_BTN_DISABLED, COLOR_MUTED
        elif b.selected:
            fill, color = COLOR_BTN_SELECTED, (255, 255, 255)
        else:
            fill, color = COLOR_BTN, COLOR_TEXT
        if b.enabled and not b.selected and b.contains(self.mouse):
            fill = (210, 205, 188)
        pygame.draw.rect(self.screen, fill, b.rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_GRID, b.rect, width=1, border_radius=6)
        surface = self.font.render(b.text, True, color)
        self.screen.blit(surface, surface.get_rect(center=b.rect.center))

    def _draw_menu(self) -> None:
        self._text("CARO 11×11 – Demo AI", (50, 40), self.font_title)
        self._text("Chế độ chơi", (50, 98), self.font_h)
        for title, x, y in self._title_at:
            self._text(title, (x, y), self.font_h)
        if self.message:
            for i, line in enumerate(wrap_text(self.font_small, self.message, 240)):
                self._text(line, (720, 585 + i * 20), self.font_small, COLOR_X)
        self._text("Luật Việt Nam: đủ 5 quân, không bị chặn hai đầu.", (50, 640), self.font_small, COLOR_MUTED)

    def _draw_board(self, game, highlight: list[int], last_move: int | None) -> None:
        ox, oy = BOARD_ORIGIN
        size = game.size
        pygame.draw.rect(self.screen, COLOR_BOARD, (ox, oy, size * CELL, size * CELL))
        hover = self.cell_at(self.mouse)
        human_turn = self.session is not None and self.replay is None and self.session.is_human_turn
        if human_turn and hover is not None and hover in game.legal_actions():
            r, c = game.action_to_coord(hover)
            pygame.draw.rect(self.screen, COLOR_HOVER, (ox + c * CELL, oy + r * CELL, CELL, CELL))
        for action in highlight:
            r, c = game.action_to_coord(action)
            pygame.draw.rect(self.screen, COLOR_WIN, (ox + c * CELL, oy + r * CELL, CELL, CELL))
        if last_move is not None:
            r, c = game.action_to_coord(last_move)
            pygame.draw.rect(self.screen, COLOR_LAST, (ox + c * CELL, oy + r * CELL, CELL, CELL))
        for i in range(size + 1):
            pygame.draw.line(self.screen, COLOR_GRID, (ox, oy + i * CELL), (ox + size * CELL, oy + i * CELL))
            pygame.draw.line(self.screen, COLOR_GRID, (ox + i * CELL, oy), (ox + i * CELL, oy + size * CELL))
        for i in range(size):
            label = str(i + 1)
            w = self.font_small.size(label)[0]
            self._text(label, (ox + i * CELL + (CELL - w) // 2, oy - 24), self.font_small, COLOR_MUTED)
            self._text(label, (ox - 12 - w // 2 - 6, oy + i * CELL + 16), self.font_small, COLOR_MUTED)
        pad = 12
        for r in range(size):
            for c in range(size):
                piece = int(game.board[r, c])
                if piece == 0:
                    continue
                x0, y0 = ox + c * CELL, oy + r * CELL
                if piece == BLACK:
                    pygame.draw.line(self.screen, COLOR_X, (x0 + pad, y0 + pad), (x0 + CELL - pad, y0 + CELL - pad), 5)
                    pygame.draw.line(self.screen, COLOR_X, (x0 + CELL - pad, y0 + pad), (x0 + pad, y0 + CELL - pad), 5)
                else:
                    pygame.draw.circle(self.screen, COLOR_O, (x0 + CELL // 2, y0 + CELL // 2), CELL // 2 - pad + 1, 5)

    def _who(self, player: int) -> str:
        assert self.session is not None
        p = self.session.participants[player]
        return "Người" if p.is_human else get_level(int(p.level)).title

    def _status_line(self) -> str:
        s = self.session
        assert s is not None
        if self.replay is not None:
            return f"Đang xem lại: nước {self.replay.index}/{self.replay.total}"
        if s.is_over:
            return f"Kết thúc: {s.result_text()}"
        name = PLAYER_NAMES[s.game.current_player]
        if s.is_human_turn:
            return f"Đến lượt bạn ({name})"
        if self.paused and not s.has_human:
            return "Tạm dừng"
        return f"AI ({name} – {self._who(s.game.current_player)}) đang suy nghĩ…"

    def _panel_record(self) -> MoveRecord | None:
        s = self.session
        assert s is not None
        if self.replay is not None:
            return self.replay.record
        for record in reversed(s.moves):
            if not record.by_human:
                return record
        return None

    def _draw_game(self) -> None:
        s = self.session
        assert s is not None
        game = self.replay.game if self.replay is not None else s.game
        highlight = find_win_line(game)
        self._draw_board(game, highlight, game.last_move)
        self._text(self._status_line(), (BOARD_ORIGIN[0], 18), self.font_big)

        x, y = PANEL_X, 70
        pygame.draw.rect(self.screen, COLOR_PANEL, (x, y, PANEL_W, 310), border_radius=8)
        pygame.draw.rect(self.screen, COLOR_GRID, (x, y, PANEL_W, 310), width=1, border_radius=8)
        px, py = x + 12, y + 10
        py += self._text(f"X: {self._who(BLACK)}", (px, py), self.font, COLOR_X) + 2
        py += self._text(f"O: {self._who(WHITE)}", (px, py), self.font, COLOR_O) + 2
        py += self._text(f"Thời gian suy nghĩ: {s.budget:.1f} giây/nước".replace(".", ","), (px, py),
                         self.font_small, COLOR_MUTED) + 10
        py += self._text("AI cân nhắc", (px, py), self.font_big) + 4
        record = self._panel_record()
        if record is None:
            self._text("Chưa có nước đi của AI.", (px, py), self.font_small, COLOR_MUTED)
        else:
            r, c = game.action_to_coord(record.action)
            number = s.moves.index(record) + 1 if record in s.moves else 0
            head = f"Nước {number}: {PLAYER_NAMES[record.player]} đi hàng {r + 1}, cột {c + 1}"
            for line in wrap_text(self.font_small, head, PANEL_W - 24):
                py += self._text(line, (px, py), self.font_small) + 1
            if record.by_human:
                self._text("Nước của người chơi (không có thông số).", (px, py + 4), self.font_small, COLOR_MUTED)
            else:
                py += 4
                for line in format_stats(record.stats, record.elapsed):
                    for sub in wrap_text(self.font_small, line, PANEL_W - 24):
                        py += self._text(sub, (px, py), self.font_small) + 1
        if self.message:
            for i, line in enumerate(wrap_text(self.font_small, self.message, PANEL_W)):
                self._text(line, (x, 580 + i * 20), self.font_small, COLOR_X)
        if self.replay is not None:
            self._text("Phím: ← → Home End, Space = phát", (x, 648), self.font_small, COLOR_MUTED)


def main() -> None:
    """Mở cửa sổ và chạy vòng lặp sự kiện."""
    pygame.init()
    pygame.display.set_caption("Caro 11x11 – Demo AI (Nhóm 31)")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    app = App(screen)
    while app.running:
        dt = clock.tick(FPS) / 1000
        for event in pygame.event.get():
            app.handle_event(event)
        app.update(dt)
        app.draw()
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
