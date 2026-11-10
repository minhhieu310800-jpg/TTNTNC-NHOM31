"""Tác tử Cấp 6: RL Self-Play Actor-Critic cho Caro 11x11.

Theo Proposal mục 4–5:
- Kế thừa kiến trúc mạng PolicyValueNet (6 khối, 64 kênh) khởi tạo từ Cấp 5 (SL)
  và huấn luyện tiếp bằng Policy Gradient có baseline (Actor-Critic) qua self-play
  với pool checkpoint cũ.
- Giao diện suy luận dùng chung logic với `SLAgent` (mask ô bận, argmax policy khi
  `temperature=0` hoặc lấy mẫu theo nhiệt độ `temperature > 0`).
"""

from __future__ import annotations

from agents.supervised_policy_value_network_agent import SLAgent


class RLAgent(SLAgent):
    """Tác tử Cấp 6: Chơi từ checkpoint RL self-play."""

    name = "rl"
