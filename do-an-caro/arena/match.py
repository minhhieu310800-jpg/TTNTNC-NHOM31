"""Chơi một ván giữa hai tác tử và trả về bản ghi log của ván đó."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from arena.agent_factory import AgentSpec, build_agent
from arena.logger import to_jsonable
from game.constants import BLACK, ENGINE_VERSION, WHITE
from game.game import CaroGame


@dataclass(frozen=True)
class GameTask:
    """Mọi thông tin cần để chơi lại đúng một ván (gửi được sang tiến trình con)."""

    game_id: str
    experiment: str
    mode: str
    budget: float | None
    size: int
    rule: str
    opening_id: int
    opening_moves: tuple[int, ...]
    seed: int
    agent_x: AgentSpec
    agent_o: AgentSpec
    seed_x: int
    seed_o: int
    extra: dict[str, Any] = field(default_factory=dict)


def play_game(task: GameTask) -> dict[str, Any]:
    """Đặt khai cuộc, cho hai tác tử luân phiên đi đến hết ván.

    Tác tử nhận một bản sao của ván cờ nên không thể làm hỏng trạng thái thật.
    Nếu tác tử ném ngoại lệ hoặc trả nước bất hợp lệ thì bị xử thua, lỗi ghi vào `error`.
    """
    started = time.perf_counter()
    game = CaroGame(size=task.size, rule=task.rule)
    for action in task.opening_moves:
        game.step(action)

    agents = {
        BLACK: build_agent(task.agent_x, task.seed_x),
        WHITE: build_agent(task.agent_o, task.seed_o),
    }
    moves: list[dict[str, Any]] = []
    error: str | None = None
    forfeit_loser: int | None = None

    while not game.is_terminal():
        player = game.current_player
        agent = agents[player]
        t0 = time.perf_counter()
        try:
            action = int(agent.select_action(game.clone(), task.budget))
        except Exception as exc:  # tác tử lỗi -> xử thua, ghi lại để điều tra
            error = f"{type(exc).__name__}: {exc}"
            forfeit_loser = player
            break
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        try:
            game.step(action)
        except (ValueError, RuntimeError) as exc:
            error = f"Nước bất hợp lệ {action}: {exc}"
            forfeit_loser = player
            break
        moves.append({
            "player": player,
            "action": action,
            "time_ms": round(elapsed_ms, 3),
            "stats": to_jsonable(getattr(agent, "last_stats", {}) or {}),
        })

    result = -forfeit_loser if forfeit_loser is not None else int(game.result())
    winner = None
    if result == BLACK:
        winner = task.agent_x.label
    elif result == WHITE:
        winner = task.agent_o.label

    return {
        "game_id": task.game_id,
        "experiment": task.experiment,
        "mode": task.mode,
        "budget": task.budget,
        "engine_version": ENGINE_VERSION,
        "rule": task.rule,
        "size": task.size,
        "opening_id": task.opening_id,
        "opening_moves": list(task.opening_moves),
        "seed": task.seed,
        "agent_x": {**task.agent_x.to_dict(), "seed": task.seed_x},
        "agent_o": {**task.agent_o.to_dict(), "seed": task.seed_o},
        "moves": moves,
        "num_moves": len(task.opening_moves) + len(moves),
        "result": result,
        "winner": winner,
        "error": error,
        "duration_s": round(time.perf_counter() - started, 3),
        **task.extra,
    }
