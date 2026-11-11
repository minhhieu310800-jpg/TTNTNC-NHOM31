"""Chạy tác tử AI ở luồng nền để cửa sổ pygame không bị đứng khi AI suy nghĩ."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from agents.base import Agent
from game.game import CaroGame


@dataclass(frozen=True)
class AIResult:
    """Kết quả một lần AI chọn nước."""

    token: int
    action: int | None
    stats: dict[str, Any]
    elapsed: float
    error: str | None = None


class AIWorker:
    """Mỗi lần chỉ chạy một lượt suy nghĩ; kết quả cũ (token cũ) bị bỏ qua."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = 0
        self._result: AIResult | None = None
        self._busy = False
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    def wait(self, timeout: float | None = 2.0) -> None:
        """Đợi luồng suy nghĩ hiện tại kết thúc (nếu đang chạy)."""
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def start(self, agent: Agent, state: CaroGame, budget: float | None) -> int:
        """Bắt đầu suy nghĩ trên bản sao của ``state``; trả về token của lượt này.

        Nếu luồng của lượt trước chưa thoát hẳn (ví dụ vừa bấm Hoàn tác), đợi luồng đó
        kết thúc trước khi mở luồng mới để hai luồng không bao giờ chạy song song.
        """
        self.wait()
        snapshot = state.clone()
        with self._lock:
            self._token += 1
            token = self._token
            self._result = None
            self._busy = True
            thread = threading.Thread(
                target=self._run, args=(token, agent, snapshot, budget), daemon=True
            )
            self._thread = thread
        thread.start()
        return token

    def cancel(self, wait: bool = False, timeout: float | None = 2.0) -> None:
        """Bỏ kết quả của lượt đang chạy; tùy chọn đợi luồng nền kết thúc."""
        with self._lock:
            self._token += 1
            self._result = None
            self._busy = False
        if wait:
            self.wait(timeout=timeout)

    def poll(self) -> AIResult | None:
        """Lấy kết quả nếu đã xong (chỉ trả một lần)."""
        with self._lock:
            result, self._result = self._result, None
            if result is not None:
                self._busy = False
            return result

    def _run(self, token: int, agent: Agent, state: CaroGame, budget: float | None) -> None:
        start = time.perf_counter()
        try:
            action = int(agent.select_action(state, budget=budget))
            result = AIResult(token, action, dict(agent.last_stats), time.perf_counter() - start)
        except Exception as exc:  # tác tử lỗi không được làm sập giao diện
            result = AIResult(token, None, {}, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
        with self._lock:
            if token == self._token:
                self._result = result

