"""Shared Caro pattern detection and heuristic evaluation module.

This module provides common pattern evaluation, threat recognition, and
move ordering for:
- Level 2 Heuristic Agent (Issue #3)
- Level 3 Alpha-Beta Minimax search (Issue #6)
- Level 4 MCTS-UCT heuristic rollouts (Issue #7)

References & Academic Integrity:
- Mark Liu, *AlphaGo Simplified* (2024), Chapter 3 (Heuristic Evaluation & Threat Space).
- Victor Allis, *Searching for Solutions in Games and Artificial Intelligence* (1994).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Final

import numpy as np

from game.constants import (
    DIRECTIONS,
    EMPTY,
    RULE_FREESTYLE,
    RULE_VIETNAM,
    WIN_STREAK,
)
from game.rules import check_win_at

if TYPE_CHECKING:
    from game.game import CaroGame

# Tên các mẫu hình chuẩn trong cờ Caro / Gomoku
PATTERN_FIVE: Final[str] = "five"                    # 5 quân liên tiếp (thắng ngay)
PATTERN_OPEN_FOUR: Final[str] = "open_four"          # Bốn mở (_XXXX_): không thể cản
PATTERN_BLOCKED_FOUR: Final[str] = "blocked_four"    # Bốn đóng / chặn 1 đầu / bốn có lỗ
PATTERN_OPEN_THREE: Final[str] = "open_three"        # Ba mở (_XXX_): dọa lên bốn mở
PATTERN_BLOCKED_THREE: Final[str] = "blocked_three"  # Ba đóng / chặn 1 đầu
PATTERN_OPEN_TWO: Final[str] = "open_two"            # Hai mở (_XX_)
PATTERN_BLOCKED_TWO: Final[str] = "blocked_two"      # Hai đóng / chặn 1 đầu
PATTERN_OPEN_ONE: Final[str] = "open_one"            # Một quân có không gian phát triển

# Thang điểm đánh giá mẫu hình
PATTERN_SCORES: Final[dict[str, int]] = {
    PATTERN_FIVE: 100_000_000,
    PATTERN_OPEN_FOUR: 10_000_000,
    PATTERN_BLOCKED_FOUR: 100_000,
    PATTERN_OPEN_THREE: 50_000,
    PATTERN_BLOCKED_THREE: 1_000,
    PATTERN_OPEN_TWO: 500,
    PATTERN_BLOCKED_TWO: 50,
    PATTERN_OPEN_ONE: 10,
}

# Điểm số thứ tự ưu tiên (cho phát hiện đòn)
_RANK: Final[dict[str, int]] = {
    PATTERN_FIVE: 7,
    PATTERN_OPEN_FOUR: 6,
    PATTERN_BLOCKED_FOUR: 5,
    PATTERN_OPEN_THREE: 4,
    PATTERN_BLOCKED_THREE: 3,
    PATTERN_OPEN_TWO: 2,
    PATTERN_BLOCKED_TWO: 1,
    PATTERN_OPEN_ONE: 0,
}


def get_candidate_moves(board: np.ndarray, radius: int = 2) -> list[int]:
    """Trả về danh sách ô trống có khoảng cách Chebyshev <= radius tới quân cờ bất kỳ.

    - Nếu bàn cờ trống hoàn toàn: trả về ô trung tâm.
    - Nếu không tìm thấy ô nào: trả về toàn bộ ô hợp lệ (fallback).

    Được dùng chung cho:
    - HeuristicAgent (#3)
    - Minimax / Alpha-Beta candidate move generation (#6: 'Chỉ xét ô bán kính 2')
    """
    size = board.shape[0]
    occupied = np.argwhere(board != EMPTY)
    if len(occupied) == 0:
        center = size // 2
        return [center * size + center]

    mask = np.zeros((size, size), dtype=bool)
    for r, c in occupied:
        r_min = max(0, r - radius)
        r_max = min(size, r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(size, c + radius + 1)
        mask[r_min:r_max, c_min:c_max] = True

    candidate_mask = mask & (board == EMPTY)
    candidates = np.flatnonzero(candidate_mask.ravel()).tolist()
    if not candidates:
        return np.flatnonzero(board.ravel() == EMPTY).tolist()
    return candidates


def is_immediate_win(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    rule: str = RULE_VIETNAM,
    win_streak: int = WIN_STREAK,
) -> bool:
    """Kiểm tra nếu đặt quân của `player` tại (row, col) thì có thắng ngay lập tức không."""
    if board[row, col] != EMPTY:
        return False
    board[row, col] = player
    won = check_win_at(board, row, col, rule=rule, win_streak=win_streak)
    board[row, col] = EMPTY
    return won


def _classify_direction_line(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
    rule: str,
) -> str | None:
    """Phân loại mẫu hình mạnh nhất chứa ô (row, col) theo một hướng (dr, dc).

    Giả định: ô (row, col) đã được gán giá trị `player`.
    """
    size_r, size_c = board.shape
    opponent = -player

    # 1. Đếm quân player liên tiếp theo chiều dương (+dr, +dc)
    len_pos = 0
    r_pos, c_pos = row + dr, col + dc
    while 0 <= r_pos < size_r and 0 <= c_pos < size_c and board[r_pos, c_pos] == player:
        len_pos += 1
        r_pos += dr
        c_pos += dc

    if 0 <= r_pos < size_r and 0 <= c_pos < size_c:
        end_pos_val = int(board[r_pos, c_pos])
    else:
        end_pos_val = -99  # Biên bàn

    # 2. Đếm quân player liên tiếp theo chiều âm (-dr, -dc)
    len_neg = 0
    r_neg, c_neg = row - dr, col - dc
    while 0 <= r_neg < size_r and 0 <= c_neg < size_c and board[r_neg, c_neg] == player:
        len_neg += 1
        r_neg -= dr
        c_neg -= dc

    if 0 <= r_neg < size_r and 0 <= c_neg < size_c:
        end_neg_val = int(board[r_neg, c_neg])
    else:
        end_neg_val = -99  # Biên bàn

    consecutive = 1 + len_pos + len_neg

    # A. 5 quân trở lên
    if consecutive >= WIN_STREAK:
        if rule == RULE_FREESTYLE:
            return PATTERN_FIVE
        if not (end_pos_val == opponent and end_neg_val == opponent):
            return PATTERN_FIVE
        # Bị chặn cả 2 đầu trong luật Việt Nam
        return None

    # B. 4 quân liên tiếp
    if consecutive == 4:
        # Cả 2 đầu trống -> Bốn mở (không thể cản)
        if end_pos_val == EMPTY and end_neg_val == EMPTY:
            return PATTERN_OPEN_FOUR
        # Một đầu trống
        if end_pos_val == EMPTY:
            next_r, next_c = r_pos + dr, c_pos + dc
            next_val = int(board[next_r, next_c]) if (0 <= next_r < size_r and 0 <= next_c < size_c) else -99
            if rule == RULE_VIETNAM and end_neg_val == opponent and next_val == opponent:
                return None
            return PATTERN_BLOCKED_FOUR
        if end_neg_val == EMPTY:
            next_r, next_c = r_neg - dr, c_neg - dc
            next_val = int(board[next_r, next_c]) if (0 <= next_r < size_r and 0 <= next_c < size_c) else -99
            if rule == RULE_VIETNAM and end_pos_val == opponent and next_val == opponent:
                return None
            return PATTERN_BLOCKED_FOUR
        return None

    # C. 3 quân liên tiếp
    if consecutive == 3:
        if end_pos_val == EMPTY and end_neg_val == EMPTY:
            # Kiểm tra xem có khoảng trống để mở rộng thành bốn mở không
            has_pos_space = (0 <= r_pos + dr < size_r and 0 <= c_pos + dc < size_c and board[r_pos + dr, c_pos + dc] == EMPTY)
            has_neg_space = (0 <= r_neg - dr < size_r and 0 <= c_neg - dc < size_c and board[r_neg - dr, c_neg - dc] == EMPTY)
            if has_pos_space or has_neg_space:
                return PATTERN_OPEN_THREE
            return PATTERN_BLOCKED_THREE
        if end_pos_val == EMPTY:
            has_pos_space = (0 <= r_pos + dr < size_r and 0 <= c_pos + dc < size_c and board[r_pos + dr, c_pos + dc] != opponent)
            return PATTERN_BLOCKED_THREE if has_pos_space else None
        if end_neg_val == EMPTY:
            has_neg_space = (0 <= r_neg - dr < size_r and 0 <= c_neg - dc < size_c and board[r_neg - dr, c_neg - dc] != opponent)
            return PATTERN_BLOCKED_THREE if has_neg_space else None
        return None

    # D. 2 quân liên tiếp
    if consecutive == 2:
        if end_pos_val == EMPTY and end_neg_val == EMPTY:
            return PATTERN_OPEN_TWO
        if end_pos_val == EMPTY or end_neg_val == EMPTY:
            return PATTERN_BLOCKED_TWO
        return None

    # E. 1 quân đơn lẻ
    if consecutive == 1:
        if end_pos_val == EMPTY and end_neg_val == EMPTY:
            return PATTERN_OPEN_ONE
        return None

    return None


def detect_patterns_at(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    rule: str = RULE_VIETNAM,
) -> dict[str, int]:
    """Phát hiện các mẫu hình được tạo thành tại ô (row, col) theo 4 hướng.

    Giả định: ô (row, col) đã chứa quân `player` (hoặc tạm gán trước khi gọi hàm).

    Returns:
        dict ánh xạ từ tên mẫu hình sang số lượng xuất hiện (ở các hướng khác nhau).
    """
    counts: dict[str, int] = {}
    for dr, dc in DIRECTIONS:
        pat = _classify_direction_line(board, row, col, dr, dc, player, rule)
        if pat is not None:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def is_open_four_at(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    rule: str = RULE_VIETNAM,
) -> bool:
    """Kiểm tra nước đi tại (row, col) có tạo thành Bốn Mở (_XXXX_) hay không."""
    if board[row, col] != EMPTY:
        return False
    board[row, col] = player
    counts = detect_patterns_at(board, row, col, player, rule)
    board[row, col] = EMPTY
    return counts.get(PATTERN_OPEN_FOUR, 0) > 0


def is_fork_at(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    rule: str = RULE_VIETNAM,
) -> bool:
    """Kiểm tra nước đi tại (row, col) có tạo ra đòn kép (Fork) hay không.

    Đòn kép bao gồm:
    - 3-3: Tạo ít nhất 2 Ba mở
    - 4-3: Tạo ít nhất 1 Bốn (mở hoặc đóng) và 1 Ba mở
    - 4-4: Tạo ít nhất 2 Bốn (mở hoặc đóng)
    """
    if board[row, col] != EMPTY:
        return False
    board[row, col] = player
    counts = detect_patterns_at(board, row, col, player, rule)
    board[row, col] = EMPTY

    fours = counts.get(PATTERN_OPEN_FOUR, 0) + counts.get(PATTERN_BLOCKED_FOUR, 0)
    open_threes = counts.get(PATTERN_OPEN_THREE, 0)

    if open_threes >= 2:
        return True
    if fours >= 1 and open_threes >= 1:
        return True
    if fours >= 2:
        return True
    return False


def score_point_patterns(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    rule: str = RULE_VIETNAM,
) -> float:
    """Tính điểm mẫu hình tấn công nếu `player` đặt quân tại (row, col)."""
    if board[row, col] != EMPTY:
        return 0.0
    board[row, col] = player
    counts = detect_patterns_at(board, row, col, player, rule)
    board[row, col] = EMPTY

    score = 0.0
    for pat, count in counts.items():
        score += PATTERN_SCORES.get(pat, 0) * count

    # Thưởng đòn kép
    fours = counts.get(PATTERN_OPEN_FOUR, 0) + counts.get(PATTERN_BLOCKED_FOUR, 0)
    open_threes = counts.get(PATTERN_OPEN_THREE, 0)
    if open_threes >= 2 or (fours >= 1 and open_threes >= 1) or fours >= 2:
        score += 200_000

    return score


def evaluate_move(
    board: np.ndarray,
    action: int,
    player: int,
    rule: str = RULE_VIETNAM,
    defense_weight: float = 1.0,
) -> float:
    """Đánh giá toàn diện một nước đi kết hợp cả tấn công, phòng thủ và vị trí.

    Args:
        board: Ma trận bàn cờ.
        action: Chỉ số ô (0..size*size-1).
        player: Người chơi hiện tại (1 hoặc -1).
        rule: "vietnam" hoặc "freestyle".
        defense_weight: Trọng số phòng thủ (mặc định: 1.0).

    Returns:
        Điểm số tổng hợp của nước đi.
    """
    size = board.shape[0]
    row, col = action // size, action % size

    # Điểm tấn công của mình
    attack_score = score_point_patterns(board, row, col, player, rule)

    # Điểm phòng thủ: đối thủ sẽ nhận được bao nhiêu điểm nếu đặt vào ô này
    defense_score = score_point_patterns(board, row, col, -player, rule)

    # Điểm ưu tiên kiểm soát trung tâm bàn cờ
    center = size // 2
    dist_to_center = abs(row - center) + abs(col - center)
    center_bonus = float(size - dist_to_center)

    return attack_score + defense_weight * defense_score + center_bonus


def order_moves_by_heuristic(
    game: CaroGame,
    player: int,
    radius: int = 2,
    defense_weight: float = 1.0,
) -> list[tuple[int, float]]:
    """Sắp xếp các nước đi ứng viên theo điểm heuristic từ cao xuống thấp.

    Phục vụ trực tiếp cho Alpha-Beta search với Move Ordering (Issue #6).
    """
    candidates = get_candidate_moves(game.board, radius=radius)
    scored_moves = [
        (action, evaluate_move(game.board, action, player, rule=game.rule, defense_weight=defense_weight))
        for action in candidates
    ]
    scored_moves.sort(key=lambda item: item[1], reverse=True)
    return scored_moves


def evaluate_board(
    board: np.ndarray,
    player: int,
    rule: str = RULE_VIETNAM,
) -> float:
    """Hàm lượng giá toàn bộ trạng thái bàn cờ từ góc nhìn của `player`.

    Dùng cho:
    - Minimax / Alpha-Beta leaf evaluation (Issue #6).

    Returns:
        Giá trị dương nếu player ưu thế, âm nếu opponent ưu thế.
    """
    opponent = -player
    player_score = 0.0
    opponent_score = 0.0

    occupied_coords = np.argwhere(board != EMPTY)
    for r, c in occupied_coords:
        p = int(board[r, c])
        counts = detect_patterns_at(board, int(r), int(c), p, rule)
        s = sum(PATTERN_SCORES.get(pat, 0) * cnt for pat, cnt in counts.items())
        if p == player:
            player_score += s
        else:
            opponent_score += s

    return player_score - opponent_score


def heuristic_rollout_action(
    game: CaroGame,
    player: int,
    rng: random.Random | None = None,
) -> int:
    """Chọn nhanh một nước đi có căn cứ heuristic cho mô phỏng MCTS (Issue #7).

    Ưu tiên:
    1. Thắng ngay nếu có
    2. Chặn đối thủ thắng ngay
    3. Chọn trong top 3 nước có điểm heuristic cao nhất (softmax hoặc ngẫu nhiên có trọng số)
    """
    if rng is None:
        rng = random.Random()

    candidates = get_candidate_moves(game.board, radius=2)
    if not candidates:
        legal = game.legal_actions()
        return rng.choice(legal) if legal else -1

    # 1. Thắng ngay
    for action in candidates:
        r, c = game.action_to_coord(action)
        if is_immediate_win(game.board, r, c, player, rule=game.rule):
            return action

    # 2. Chặn thắng
    opponent = -player
    for action in candidates:
        r, c = game.action_to_coord(action)
        if is_immediate_win(game.board, r, c, opponent, rule=game.rule):
            return action

    # 3. Lấy top 3 nước heuristic tốt nhất
    scored = [
        (action, evaluate_move(game.board, action, player, rule=game.rule))
        for action in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_moves = [a for a, _ in scored[:min(3, len(scored))]]
    return rng.choice(top_moves)
