"""Tác tử Cấp 5: Neural Network Supervised Learning (SL) cho Caro 11x11.

Theo Proposal mục 4–5:
- Mạng ResNet nhỏ (6 khối, 64 kênh) hai đầu policy + value, học từ ván Alpha-Beta tự chơi.
- Khi thi đấu, tác tử chơi bằng argmax policy: chọn nước đi hợp lệ có xác suất cao nhất.
- Giá trị đầu ra của value ước lượng kết quả ván từ góc nhìn người đến lượt (+1 chắc thắng, -1 chắc thua).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from agents.base import Agent
from agents.network import NetEvaluator
from agents.puct_agent import cached_evaluator

if TYPE_CHECKING:
    from game.game import CaroGame


class SLAgent(Agent):
    """Tác tử Cấp 5: Chơi bằng argmax policy từ mạng policy-value đã huấn luyện SL."""

    name = "sl"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        evaluator: NetEvaluator | None = None,
        device: str = "auto",
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if evaluator is None:
            if checkpoint is None:
                raise ValueError("Cần cung cấp `checkpoint` hoặc `evaluator`.")
            evaluator = cached_evaluator(str(checkpoint), device)
        self.evaluator = evaluator
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.device = device
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.last_stats: dict[str, Any] = {}

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """Chọn nước đi dựa trên mạng nơ-ron (argmax policy đã mask ô bận).

        Args:
            state: Trạng thái bàn cờ hiện tại.
            budget: Thời gian suy nghĩ tối đa (giây). Mạng SL suy luận qua một lượt
                    forward pass nên hoàn thành trong vài mili-giây.

        Returns:
            Chỉ số ô 0..120 hợp lệ.
        """
        start = time.perf_counter()
        legals = state.legal_actions()
        if not legals:
            raise ValueError("Không còn nước đi hợp lệ.")

        # Dự đoán policy (đã mask ô bận và softmax) cùng value; chạy đều cả khi còn 1 nước để cập nhật last_stats
        probs, value = self.evaluator.predict(state)

        if self.temperature <= 1e-6:
            best_action = int(np.argmax(probs))
            if best_action not in legals:
                best_action = max(legals, key=lambda a: float(probs[a]))
            chosen_action = best_action
        else:
            legal_probs = np.array([probs[a] for a in legals], dtype=np.float64)
            temp = max(self.temperature, 1e-3)
            logits = np.log(np.maximum(legal_probs, 1e-12)) / temp
            logits -= np.max(logits)
            exp_logits = np.exp(logits)
            p = exp_logits / np.sum(exp_logits)
            chosen_action = int(self.rng.choice(legals, p=p))

        self.last_stats = {
            "value": float(value),
            "policy_prob": float(probs[chosen_action]),
            "action": chosen_action,
            "temperature": self.temperature,
            "time_used": time.perf_counter() - start,
        }
        return chosen_action
