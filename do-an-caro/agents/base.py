"""Base class for all Caro AI agents.

Follows the exact contract defined in README.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from game.game import CaroGame


class Agent:
    """Interface chung cho mọi tác tử AI của đồ án Caro 11×11."""

    name: str = "base"

    def __init__(self) -> None:
        """Khởi tạo tác tử với từ điển thống kê rỗng."""
        self.last_stats: dict[str, Any] = {}

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Trả về chỉ số ô 0–120 (hàng * 11 + cột).

        Args:
            state: Trạng thái game hiện tại (không được sửa trực tiếp, dùng state.clone() hoặc step/undo).
            budget: Thời gian suy nghĩ tối đa (giây) cho nước này; None = dùng mặc định.

        Returns:
            Chỉ số ô hành động hợp lệ (0..size*size-1).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
