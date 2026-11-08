"""Logic ván cờ của giao diện, không phụ thuộc pygame nên kiểm thử được bằng pytest.

- ``GameSession``: ván đang chơi (người/máy), lịch sử nước đi kèm ``last_stats``, undo.
- ``Replay``: xem lại ván theo từng nước, lùi/tiến và tự phát.
- ``find_win_line``: tìm dãy quân thắng để tô sáng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.base import Agent
from game.constants import BLACK, DEFAULT_BOARD_SIZE, DIRECTIONS, EMPTY, RULE_VIETNAM, WHITE
from game.game import CaroGame
from ui.levels import DEFAULT_BUDGET

PLAYER_NAMES: dict[int, str] = {BLACK: "X", WHITE: "O"}


@dataclass
class Participant:
    """Một bên chơi: người (``agent is None``) hoặc máy với một cấp AI."""

    level: int | None = None
    agent: Agent | None = None

    @property
    def is_human(self) -> bool:
        return self.agent is None


@dataclass(frozen=True)
class MoveRecord:
    """Một nước đã đi, kèm thông số tác tử cân nhắc (rỗng với người chơi)."""

    player: int
    action: int
    stats: dict[str, Any] = field(default_factory=dict)
    elapsed: float | None = None
    by_human: bool = False


class GameSession:
    """Ván cờ đang diễn ra giữa hai ``Participant``."""

    def __init__(
        self,
        black: Participant,
        white: Participant,
        budget: float = DEFAULT_BUDGET,
        size: int = DEFAULT_BOARD_SIZE,
        rule: str = RULE_VIETNAM,
    ) -> None:
        self.participants: dict[int, Participant] = {BLACK: black, WHITE: white}
        self.budget = budget
        self.game = CaroGame(size=size, rule=rule)
        self.moves: list[MoveRecord] = []

    # ---- trạng thái ----
    @property
    def current(self) -> Participant:
        return self.participants[self.game.current_player]

    @property
    def is_over(self) -> bool:
        return self.game.is_terminal()

    @property
    def is_human_turn(self) -> bool:
        return not self.is_over and self.current.is_human

    @property
    def is_ai_turn(self) -> bool:
        return not self.is_over and not self.current.is_human

    @property
    def has_human(self) -> bool:
        return any(p.is_human for p in self.participants.values())

    def result_text(self) -> str:
        """Câu mô tả kết quả; rỗng nếu ván chưa kết thúc."""
        if not self.is_over:
            return ""
        winner = self.game.result()
        if winner == 0:
            return "Hòa (bàn cờ đầy)"
        return f"{PLAYER_NAMES[int(winner)]} thắng"

    # ---- hành động ----
    def play(self, action: int, stats: dict[str, Any] | None = None, elapsed: float | None = None) -> MoveRecord:
        """Đi một nước cho bên đang đến lượt; ném lỗi nếu nước không hợp lệ."""
        if self.is_over:
            raise RuntimeError("Ván cờ đã kết thúc.")
        if action not in self.game.legal_actions():
            raise ValueError(f"Nước đi {action} không hợp lệ.")
        record = MoveRecord(
            player=self.game.current_player,
            action=action,
            stats=dict(stats or {}),
            elapsed=elapsed,
            by_human=self.current.is_human,
        )
        self.game.step(action)
        self.moves.append(record)
        return record

    def can_undo(self) -> bool:
        """Chỉ cho undo khi có người chơi và còn nước để thu hồi."""
        return self.has_human and bool(self.moves)

    def undo(self) -> int:
        """Thu hồi nước đi cho đến khi lại đến lượt người; trả về số nước đã bỏ."""
        if not self.can_undo():
            return 0
        removed = 0
        while self.moves:
            self.game.undo()
            self.moves.pop()
            removed += 1
            if self.current.is_human:
                break
        return removed

    def reset(self) -> None:
        """Bắt đầu lại ván mới với cùng cấu hình."""
        self.game.reset()
        self.moves.clear()


def find_win_line(game: CaroGame) -> list[int]:
    """Dãy ô thẳng hàng của bên thắng (chứa nước cuối); rỗng nếu chưa có người thắng."""
    if game.winner not in (BLACK, WHITE) or game.last_move is None:
        return []
    row, col = game.action_to_coord(game.last_move)
    color = int(game.board[row, col])
    best: list[int] = []
    for dr, dc in DIRECTIONS:
        cells = [game.last_move]
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < game.size and 0 <= c < game.size and int(game.board[r, c]) == color:
                cells.append(game.coord_to_action(r, c))
                r, c = r + sign * dr, c + sign * dc
        if len(cells) >= game.win_streak and len(cells) > len(best):
            best = cells
    return sorted(best)


class Replay:
    """Xem lại một ván đã đi: bàn cờ tại nước thứ ``index`` (0 = bàn rỗng)."""

    def __init__(self, size: int, rule: str, moves: list[MoveRecord]) -> None:
        self.size = size
        self.rule = rule
        self.moves = list(moves)
        self.index = len(self.moves)
        self.autoplay = False
        self._clock = 0.0
        self._game: CaroGame | None = None
        self._game_index = -1

    @property
    def total(self) -> int:
        return len(self.moves)

    @property
    def game(self) -> CaroGame:
        """Bàn cờ tại vị trí hiện tại (dựng lại khi ``index`` đổi)."""
        if self._game is None or self._game_index != self.index:
            game = CaroGame(size=self.size, rule=self.rule)
            for record in self.moves[: self.index]:
                game.step(record.action)
            self._game, self._game_index = game, self.index
        return self._game

    @property
    def record(self) -> MoveRecord | None:
        """Nước đi vừa thực hiện để đến vị trí hiện tại."""
        return self.moves[self.index - 1] if self.index > 0 else None

    def goto(self, index: int) -> None:
        self.index = max(0, min(self.total, index))

    def first(self) -> None:
        self.goto(0)

    def prev(self) -> None:
        self.goto(self.index - 1)

    def next(self) -> None:
        self.goto(self.index + 1)

    def last(self) -> None:
        self.goto(self.total)

    def toggle_autoplay(self) -> None:
        """Bật/tắt tự phát; bắt đầu lại từ đầu nếu đang ở cuối ván."""
        self.autoplay = not self.autoplay
        self._clock = 0.0
        if self.autoplay and self.index >= self.total:
            self.first()

    def tick(self, dt: float, interval: float = 0.6) -> None:
        """Tiến thời gian ``dt`` giây; tự đi tới một nước mỗi ``interval`` giây."""
        if not self.autoplay:
            return
        self._clock += dt
        while self._clock >= interval and self.index < self.total:
            self._clock -= interval
            self.next()
        if self.index >= self.total:
            self.autoplay = False

