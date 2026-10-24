"""Constants for Caro (Gomoku) 11x11 game."""

from typing import Final

# Người chơi và ô cờ
BLACK: Final[int] = 1   # Quân X, đi trước
WHITE: Final[int] = -1  # Quân O, đi sau
EMPTY: Final[int] = 0   # Ô trống

# Cấu hình mặc định
DEFAULT_BOARD_SIZE: Final[int] = 11
WIN_STREAK: Final[int] = 5

# Luật chơi
RULE_VIETNAM: Final[str] = "vietnam"      # Thắng >= 5 quân, không bị chặn 2 đầu, biên không tính chặn
RULE_FREESTYLE: Final[str] = "freestyle"  # Thắng >= 5 quân liên tiếp bất kể bị chặn

# 4 hướng kiểm tra thắng: (dr, dc)
# (0, 1): Ngang, (1, 0): Dọc, (1, 1): Chéo chính \, (1, -1): Chéo phụ /
DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)

# Ký tự hiển thị bàn cờ
PLAYER_SYMBOLS: Final[dict[int, str]] = {
    BLACK: "X",
    WHITE: "O",
    EMPTY: ".",
}
