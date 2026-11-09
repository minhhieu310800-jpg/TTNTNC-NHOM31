"""Mạng policy-value dùng chung cho cấp 5, 7, 8.

Input : tensor (N, 4, S, S) float32 lấy từ `CaroGame.get_planes()`
        (quân mình, quân đối thủ, nước vừa đi, lượt đi).
Output: policy_logits (N, S*S) chưa softmax; value (N,) qua tanh, trong [-1, 1],
        theo góc nhìn người đến lượt.

Kiến trúc ResNet nhỏ kiểu AlphaGo Zero: lớp đầu conv 3x3, `blocks` khối residual,
đầu policy (conv 1x1 -> fully connected) và đầu value (conv 1x1 -> FC -> tanh).
Cấp 5/6 dùng 6 khối, cấp 8 dùng 4 khối (proposal mục 4–5), cùng 64 kênh.

Tham khảo: Silver et al. (2017), Mastering the game of Go without human knowledge;
Mark Liu (2024), AlphaGo Simplified, ch. 16–21.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from game.constants import DEFAULT_BOARD_SIZE, ENGINE_VERSION

if TYPE_CHECKING:
    from game.game import CaroGame

IN_PLANES = 4
DEFAULT_BLOCKS = 6
DEFAULT_CHANNELS = 64
VALUE_HIDDEN = 64


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """ResNet hai đầu policy + value."""

    def __init__(
        self,
        blocks: int = DEFAULT_BLOCKS,
        channels: int = DEFAULT_CHANNELS,
        size: int = DEFAULT_BOARD_SIZE,
    ) -> None:
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels, "size": size}
        self.size = size
        cells = size * size
        self.stem = nn.Sequential(
            nn.Conv2d(IN_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True))
        self.policy_fc = nn.Linear(2 * cells, cells)
        self.value_conv = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True))
        self.value_fc = nn.Sequential(nn.Linear(cells, VALUE_HIDDEN), nn.ReLU(inplace=True), nn.Linear(VALUE_HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        policy_logits = self.policy_fc(self.policy_conv(h).flatten(1))
        value = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1)
        return policy_logits, value


# ---------------------------------------------------------------- mã hóa ván cờ

def encode_game(game: CaroGame) -> tuple[np.ndarray, np.ndarray]:
    """(planes (4,S,S) float32, mask (S*S,) bool: ô còn trống)."""
    return game.get_planes(), (game.board.ravel() == 0)


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax theo hàng, ô bận nhận xác suất 0 rồi chuẩn hóa lại."""
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- suy luận

def pick_device(name: str = "auto") -> torch.device:
    """'auto' chọn cuda nếu có, ngược lại cpu (mps chậm với batch nhỏ nên không tự chọn)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class NetEvaluator:
    """Bọc mạng để PUCT gọi: evaluate(planes, masks) -> (priors, values) dạng NumPy."""

    def __init__(self, model: PolicyValueNet, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def evaluate(self, planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).to(self.device)
        logits, values = self.model(x)
        priors = masked_softmax(logits.float().cpu().numpy(), masks)
        return priors, values.float().cpu().numpy()

    def predict_batch(self, planes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dự đoán cho một batch thế cờ (planes (N,4,S,S), masks (N,S*S) bool)."""
        return self.evaluate(planes, masks)

    def predict(self, game: CaroGame) -> tuple[np.ndarray, float]:
        """Xác suất 121 ô (đã mask) và value của một ván, theo góc nhìn người đến lượt."""
        planes, mask = encode_game(game)
        priors, values = self.evaluate(planes[None], mask[None])
        return priors[0], float(values[0])


# ---------------------------------------------------------------- checkpoint

def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    epoch: int | None = None,
    val_metrics: dict[str, Any] | None = None,
    seed: int | None = None,
    **extra: Any,
) -> Path:
    """Lưu {"model_state", "config", "epoch", "val_metrics", "seed", "engine_version", ...}.

    Ghi file tạm rồi đổi tên nguyên tử để không hỏng checkpoint khi bị ngắt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dict(model.config),
        "engine_version": ENGINE_VERSION,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "seed": seed,
        **extra,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[PolicyValueNet, dict[str, Any]]:
    """Dựng đúng mạng theo `config` trong checkpoint và nạp trọng số."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PolicyValueNet(**payload["config"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "model_state"}
    return model, meta


def load_network(path: str | Path, device: str | torch.device = "cpu") -> tuple[PolicyValueNet, dict[str, Any]]:
    """Tải mạng và metadata từ checkpoint (alias của load_checkpoint theo chuẩn issue #10)."""
    return load_checkpoint(path, device)


def load_evaluator(path: str | Path, device: str = "auto") -> NetEvaluator:
    dev = pick_device(device)
    model, _ = load_checkpoint(path, dev)
    return NetEvaluator(model, dev)
