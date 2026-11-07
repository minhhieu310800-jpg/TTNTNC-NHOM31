"""PUCT: tìm kiếm cây dùng mạng policy-value (cấp 7 và cấp 8).

Khác MCTS-UCT thuần (cấp 4):
- Chọn nước theo Q + U, với U = c_puct * P * sqrt(N_cha) / (1 + N_con); P là xác suất của mạng.
- Không rollout: giá trị lá lấy từ đầu value của mạng (hoặc kết quả thật nếu ván đã kết thúc).

Cây được tách thành `select_leaf` (đi xuống tới lá, trả về input cho mạng) và
`backup` (mở rộng lá, cập nhật thống kê) để self-play gom lá của nhiều ván thành
một batch suy luận (proposal mục 5).

Tham khảo: Silver et al. (2016, 2017); Mark Liu (2024), AlphaGo Simplified, ch. 16–21.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from agents.base import Agent
from agents.network import encode_game, load_evaluator

if TYPE_CHECKING:
    from game.game import CaroGame

DEFAULT_C_PUCT = 1.5
DEFAULT_SIMULATIONS = 400       # chế độ B: PUCT 400 mô phỏng (proposal mục 6)


class Evaluator(Protocol):
    def evaluate(self, planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class UniformEvaluator:
    """Mạng giả: xác suất đều trên ô trống, value 0. Chỉ dùng để kiểm thử."""

    def evaluate(self, planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = masks.astype(np.float64)
        return m / m.sum(axis=1, keepdims=True), np.zeros(len(masks))


class Node:
    """Nút cây; thống kê của các con lưu dạng mảng để chọn nước bằng NumPy."""

    __slots__ = ("to_play", "actions", "P", "N", "W", "children", "total")

    def __init__(self, to_play: int) -> None:
        self.to_play = to_play          # người đến lượt tại nút này
        self.actions: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None  # tổng giá trị theo góc nhìn `to_play`
        self.children: dict[int, Node] = {}
        self.total = 0

    @property
    def expanded(self) -> bool:
        return self.actions is not None

    def expand(self, priors: np.ndarray, mask: np.ndarray) -> None:
        self.actions = np.flatnonzero(mask)
        p = priors[self.actions].astype(np.float64)
        s = p.sum()
        self.P = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        self.N = np.zeros(len(p))
        self.W = np.zeros(len(p))

    def select(self, c_puct: float) -> int:
        q = np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N > 0)
        u = c_puct * self.P * math.sqrt(max(self.total, 1)) / (1.0 + self.N)
        return int(np.argmax(q + u))


@dataclass
class Leaf:
    node: Node
    path: list[tuple[Node, int]]
    value: float | None = None          # có giá trị nếu lá là thế kết thúc
    planes: np.ndarray | None = None
    mask: np.ndarray | None = None


class PUCTTree:
    def __init__(
        self,
        to_play: int,
        c_puct: float = DEFAULT_C_PUCT,
        dirichlet: tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.root = Node(to_play)
        self.c_puct = c_puct
        self.dirichlet = dirichlet       # (alpha, epsilon) cho self-play; None khi thi đấu
        self.rng = rng or np.random.default_rng()
        self._root_noised = False

    def _maybe_add_noise(self) -> None:
        if self.dirichlet and not self._root_noised and self.root.expanded:
            alpha, eps = self.dirichlet
            noise = self.rng.dirichlet([alpha] * len(self.root.P))
            self.root.P = (1 - eps) * self.root.P + eps * noise
            self._root_noised = True

    def select_leaf(self, game: CaroGame) -> Leaf:
        """Đi xuống tới lá; `game` được trả về đúng trạng thái ban đầu."""
        self._maybe_add_noise()
        node = self.root
        path: list[tuple[Node, int]] = []
        steps = 0
        try:
            while node.expanded and not game.is_terminal():
                idx = node.select(self.c_puct)
                game.step(int(node.actions[idx]))
                steps += 1
                path.append((node, idx))
                child = node.children.get(idx)
                if child is None:
                    child = Node(-node.to_play)
                    node.children[idx] = child
                node = child
            if game.is_terminal():
                # Người vừa đi đã thắng (hoặc hòa) -> người đến lượt ở lá thua.
                return Leaf(node, path, value=0.0 if game.result() == 0 else -1.0)
            planes, mask = encode_game(game)
            return Leaf(node, path, planes=planes, mask=mask)
        finally:
            for _ in range(steps):
                game.undo()

    def backup(self, leaf: Leaf, prior: np.ndarray | None = None, value: float | None = None) -> None:
        """Mở rộng lá (nếu cần) và cập nhật thống kê từ lá lên gốc."""
        if leaf.value is None:
            leaf.node.expand(prior, leaf.mask)
            v = float(value)
        else:
            v = leaf.value
        val = -v  # giá trị theo góc nhìn người đến lượt ở nút cha của lá
        for parent, idx in reversed(leaf.path):
            parent.N[idx] += 1
            parent.W[idx] += val
            parent.total += 1
            val = -val

    def visit_policy(self, num_cells: int) -> np.ndarray:
        pi = np.zeros(num_cells, dtype=np.float32)
        if self.root.total > 0:
            pi[self.root.actions] = self.root.N / self.root.N.sum()
        elif self.root.expanded:
            pi[self.root.actions] = self.root.P
        return pi

    def advance(self, action: int) -> None:
        """Đi nước `action`: giữ lại cây con để dùng tiếp."""
        root = self.root
        child = None
        if root.expanded:
            idx = int(np.searchsorted(root.actions, action))
            if idx < len(root.actions) and root.actions[idx] == action:
                child = root.children.get(idx)
        self.root = child if child is not None else Node(-root.to_play)
        self._root_noised = False


def run_search(
    tree: PUCTTree,
    game: CaroGame,
    evaluator: Evaluator,
    simulations: int,
    deadline: float | None = None,
) -> int:
    """Chạy PUCT cho một ván; trả về số mô phỏng đã chạy."""
    sims = 0
    while True:
        if deadline is None:
            if sims >= simulations:
                break
        elif sims > 1 and time.perf_counter() >= deadline:
            break
        leaf = tree.select_leaf(game)
        if leaf.value is None:
            priors, values = evaluator.evaluate(leaf.planes[None], leaf.mask[None])
            tree.backup(leaf, priors[0], values[0])
        else:
            tree.backup(leaf)
        sims += 1
    return sims


_EVALUATOR_CACHE: dict[tuple[str, str, float], Any] = {}


def cached_evaluator(checkpoint: str, device: str) -> Any:
    """Nạp checkpoint một lần cho mỗi tiến trình (arena tạo tác tử mới mỗi ván)."""
    path = str(Path(checkpoint).resolve())
    key = (path, device, Path(path).stat().st_mtime)
    if key not in _EVALUATOR_CACHE:
        _EVALUATOR_CACHE[key] = load_evaluator(path, device)
    return _EVALUATOR_CACHE[key]


class PUCTAgent(Agent):
    """Tác tử PUCT: cấp 7 (mạng cấp 5/6) hoặc cấp 8 (mạng AlphaZero-inspired)."""

    name = "puct"

    def __init__(
        self,
        checkpoint: str | None = None,
        evaluator: Evaluator | None = None,
        simulations: int = DEFAULT_SIMULATIONS,
        c_puct: float = DEFAULT_C_PUCT,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if evaluator is None:
            if checkpoint is None:
                raise ValueError("Cần `checkpoint` hoặc `evaluator`.")
            evaluator = cached_evaluator(checkpoint, device)
        self.evaluator = evaluator
        self.checkpoint = checkpoint
        self.simulations = simulations
        self.c_puct = c_puct

    def select_action(self, state: CaroGame, budget: float | None = None) -> int:
        """budget=None: `simulations` mô phỏng; budget=số: chạy theo giây."""
        start = time.perf_counter()
        game = state.clone()
        tree = PUCTTree(game.current_player, self.c_puct)
        deadline = None if budget is None else start + budget
        sims = run_search(tree, game, self.evaluator, self.simulations, deadline)
        root = tree.root
        if root.total > 0:
            best = int(np.argmax(root.N))
            value = float(root.W[best] / root.N[best])
            visits = int(root.N[best])
        else:
            best = int(np.argmax(root.P))
            value, visits = None, 0
        self.last_stats = {
            "sims": sims,
            "best_visits": visits,
            "value": value,
            "prior_best": float(root.P[best]),
            "time_used": time.perf_counter() - start,
        }
        return int(root.actions[best])
