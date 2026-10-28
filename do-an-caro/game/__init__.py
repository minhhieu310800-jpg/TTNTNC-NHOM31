"""Caro Game Package."""

from game.constants import (
    BLACK,
    DEFAULT_BOARD_SIZE,
    EMPTY,
    ENGINE_VERSION,
    PLAYER_SYMBOLS,
    RULE_FREESTYLE,
    RULE_VIETNAM,
    WHITE,
    WIN_STREAK,
)
from game.game import CaroGame
from game import patterns
from game.rules import check_win_at, check_win_global

__all__ = [
    "CaroGame",
    "BLACK",
    "WHITE",
    "EMPTY",
    "PLAYER_SYMBOLS",
    "DEFAULT_BOARD_SIZE",
    "WIN_STREAK",
    "RULE_VIETNAM",
    "RULE_FREESTYLE",
    "ENGINE_VERSION",
    "patterns",
    "check_win_at",
    "check_win_global",
]
