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
PATTERN_BLOCKED_FOUR: Final[str] = "blocked_four"    # Bốn đóng / chặn 1 đầu / bốn có khe (XXX_X, XX_XX, X_XXX)
PATTERN_OPEN_THREE: Final[str] = "open_three"        # Ba mở (_XXX_ / _XX_X_ / _X_XX_): dọa lên bốn mở
PATTERN_BLOCKED_THREE: Final[str] = "blocked_three"  # Ba đóng / chặn 1 đầu / ba có khe
PATTERN_OPEN_TWO: Final[str] = "open_two"            # Hai mở (_XX_ / _X_X_)
PATTERN_BLOCKED_TWO: Final[str] = "blocked_two"      # Hai đóng / chặn 1 đầu
PATTERN_OPEN_ONE: Final[str] = "open_one"            # Một quân đơn lẻ

# Thang điểm đánh giá mẫu hình chuẩn
PATTERN_SCORES: Final[dict[str, int]] = {
    PATTERN_FIVE: 100_000_000,
    PATTERN_OPEN_FOUR: 10_000_000,
    PATTERN_BLOCKED_FOUR: 100_000,
    PATTERN_OPEN_THREE: 50_000,
    PATTERN_BLOCKED_THREE: 1_000,
    PATTERN_OPEN_TWO: 500,
    PATTERN_BLOCKED_TWO: 50,
    PATTERN_OPEN_ONE: 0,
}

# Điểm số thứ tự ưu tiên (cho phát hiện mẫu hình mạnh nhất)
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


def _classify_and_collect_stones(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
    rule: str = RULE_VIETNAM,
) -> tuple[str | None, list[tuple[int, int]]]:
    """Phân loại mẫu hình mạnh nhất chứa ô (row, col) theo hướng (dr, dc)
    và trả về danh sách toạ độ các quân cờ tạo nên mẫu hình đó.

    Hỗ trợ nhận diện cả mẫu hình liền và mẫu hình có khe:
    - Bốn có khe: XXX_X, XX_XX, X_XXX -> PATTERN_BLOCKED_FOUR
    - Ba có khe: _XX_X_, _X_XX_ -> PATTERN_OPEN_THREE

    Giả định: ô (row, col) đã chứa quân `player`.
    """
    size_r, size_c = board.shape
    opponent = -player

    # Trích xuất đoạn 11 ô từ k = -5 đến k = +5 quanh (row, col)
    # k = 0 nằm tại chỉ số 5
    cells: list[int] = []
    coords: list[tuple[int, int] | None] = []

    for k in range(-5, 6):
        rk = row + k * dr
        ck = col + k * dc
        if 0 <= rk < size_r and 0 <= ck < size_c:
            cells.append(int(board[rk, ck]))
            coords.append((rk, ck))
        else:
            cells.append(-99)  # Biên bàn cờ
            coords.append(None)

    best_pattern: str | None = None
    best_rank = -1
    best_stones: list[tuple[int, int]] = []

    # 1. Kiểm tra 5 quân liên tiếp (PATTERN_FIVE)
    for start in range(1, 6):
        window = cells[start : start + 5]
        if all(c == player for c in window):
            before = cells[start - 1] if start > 0 else -99
            after = cells[start + 5] if start + 5 < 11 else -99
            if rule == RULE_VIETNAM and before == opponent and after == opponent:
                continue  # Bị chặn cả 2 đầu trong luật Việt Nam
            stones = [coords[start + j] for j in range(5) if coords[start + j] is not None]
            if _RANK[PATTERN_FIVE] > best_rank:
                best_rank = _RANK[PATTERN_FIVE]
                best_pattern = PATTERN_FIVE
                best_stones = stones

    # 2. Kiểm tra Bốn mở (PATTERN_OPEN_FOUR): cửa sổ 6 ô _XXXX_
    for start in range(0, 6):
        window = cells[start : start + 6]
        if window[0] == EMPTY and window[5] == EMPTY:
            middle = window[1:5]
            if all(c == player for c in middle):
                stones = [coords[start + 1 + j] for j in range(4) if coords[start + 1 + j] is not None]
                if _RANK[PATTERN_OPEN_FOUR] > best_rank:
                    best_rank = _RANK[PATTERN_OPEN_FOUR]
                    best_pattern = PATTERN_OPEN_FOUR
                    best_stones = stones

    # 3. Kiểm tra Bốn chặn / Bốn có khe (PATTERN_BLOCKED_FOUR):
    # Cửa sổ 5 ô có 4 quân player và 1 ô EMPTY (ví dụ: XXX_X, XX_XX, X_XXX, XXXX_, _XXXX)
    for start in range(1, 6):
        window = cells[start : start + 5]
        if all(c in (player, EMPTY) for c in window):
            count_p = sum(1 for c in window if c == player)
            count_e = sum(1 for c in window if c == EMPTY)
            if count_p == 4 and count_e == 1:
                before = cells[start - 1] if start > 0 else -99
                after = cells[start + 5] if start + 5 < 11 else -99
                # Nếu luật VN và ô trống nằm ở rìa và bị chặn 2 đầu khi đặt vào
                if rule == RULE_VIETNAM and before == opponent and after == opponent:
                    continue
                stones = [
                    coords[start + j]
                    for j in range(5)
                    if coords[start + j] is not None and window[j] == player
                ]
                if _RANK[PATTERN_BLOCKED_FOUR] > best_rank:
                    best_rank = _RANK[PATTERN_BLOCKED_FOUR]
                    best_pattern = PATTERN_BLOCKED_FOUR
                    best_stones = stones

    # 4. Kiểm tra Ba mở (PATTERN_OPEN_THREE):
    # Cửa sổ 6 ô hở 2 đầu _XXX__ / __XXX_ / _XX_X_ / _X_XX_
    for start in range(0, 6):
        window = cells[start : start + 6]
        if window[0] == EMPTY and window[5] == EMPTY:
            middle = window[1:5]
            if all(c in (player, EMPTY) for c in middle):
                count_p = sum(1 for c in middle if c == player)
                count_e = sum(1 for c in middle if c == EMPTY)
                if count_p == 3 and count_e == 1:
                    stones = [
                        coords[start + 1 + j]
                        for j in range(4)
                        if coords[start + 1 + j] is not None and middle[j] == player
                    ]
                    if _RANK[PATTERN_OPEN_THREE] > best_rank:
                        best_rank = _RANK[PATTERN_OPEN_THREE]
                        best_pattern = PATTERN_OPEN_THREE
                        best_stones = stones

    # 5. Kiểm tra Ba chặn (PATTERN_BLOCKED_THREE): cửa sổ 5 ô có 3 quân player và 2 ô EMPTY
    for start in range(1, 6):
        window = cells[start : start + 5]
        if all(c in (player, EMPTY) for c in window):
            count_p = sum(1 for c in window if c == player)
            count_e = sum(1 for c in window if c == EMPTY)
            if count_p == 3 and count_e == 2:
                stones = [
                    coords[start + j]
                    for j in range(5)
                    if coords[start + j] is not None and window[j] == player
                ]
                if _RANK[PATTERN_BLOCKED_THREE] > best_rank:
                    best_rank = _RANK[PATTERN_BLOCKED_THREE]
                    best_pattern = PATTERN_BLOCKED_THREE
                    best_stones = stones

    # 6. Kiểm tra Hai mở (PATTERN_OPEN_TWO): cửa sổ 6 ô hở 2 đầu có 2 quân player và 2 ô EMPTY
    for start in range(0, 6):
        window = cells[start : start + 6]
        if window[0] == EMPTY and window[5] == EMPTY:
            middle = window[1:5]
            if all(c in (player, EMPTY) for c in middle):
                count_p = sum(1 for c in middle if c == player)
                count_e = sum(1 for c in middle if c == EMPTY)
                if count_p == 2 and count_e == 2:
                    stones = [
                        coords[start + 1 + j]
                        for j in range(4)
                        if coords[start + 1 + j] is not None and middle[j] == player
                    ]
                    if _RANK[PATTERN_OPEN_TWO] > best_rank:
                        best_rank = _RANK[PATTERN_OPEN_TWO]
                        best_pattern = PATTERN_OPEN_TWO
                        best_stones = stones

    # 7. Kiểm tra Hai chặn (PATTERN_BLOCKED_TWO): cửa sổ 5 ô có 2 quân player và 3 ô EMPTY
    for start in range(1, 6):
        window = cells[start : start + 5]
        if all(c in (player, EMPTY) for c in window):
            count_p = sum(1 for c in window if c == player)
            count_e = sum(1 for c in window if c == EMPTY)
            if count_p == 2 and count_e == 3:
                stones = [
                    coords[start + j]
                    for j in range(5)
                    if coords[start + j] is not None and window[j] == player
                ]
                if _RANK[PATTERN_BLOCKED_TWO] > best_rank:
                    best_rank = _RANK[PATTERN_BLOCKED_TWO]
                    best_pattern = PATTERN_BLOCKED_TWO
                    best_stones = stones

    return best_pattern, best_stones


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
    pat, _ = _classify_and_collect_stones(board, row, col, dr, dc, player, rule)
    return pat


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
    - 3-3: Tạo ít nhất 2 Ba mở (kể cả ba liền _XXX_ và ba có khe _XX_X_, _X_XX_)
    - 4-3: Tạo ít nhất 1 Bốn (mở hoặc đóng/khe) và 1 Ba mở
    - 4-4: Tạo ít nhất 2 Bốn (mở hoặc đóng/khe)
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

    Mỗi hướng chỉ tính điểm đúng 1 lần cho mỗi chuỗi mẫu hình (thay vì tính n lần
    cho từng quân trong chuỗi n quân).

    - 1 ba mở: 50.000
    - 1 bốn chặn: 100.000

    Dùng cho:
    - Minimax / Alpha-Beta leaf evaluation (Issue #6).

    Returns:
        Giá trị dương nếu player ưu thế, âm nếu opponent ưu thế.
    """
    player_score = 0.0
    opponent_score = 0.0
    size_r, size_c = board.shape

    for dr, dc in DIRECTIONS:
        visited = np.zeros((size_r, size_c), dtype=bool)
        for r in range(size_r):
            for c in range(size_c):
                if board[r, c] == EMPTY or visited[r, c]:
                    continue
                p = int(board[r, c])
                pat, stones = _classify_and_collect_stones(board, r, c, dr, dc, p, rule)
                if pat is not None:
                    score = float(PATTERN_SCORES.get(pat, 0))
                    if p == player:
                        player_score += score
                    else:
                        opponent_score += score

                # Đánh dấu các quân cờ thuộc mẫu hình này đã được xét theo hướng (dr, dc)
                for pr, pc in stones:
                    visited[pr, pc] = True
                visited[r, c] = True

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
    3. Chọn trong top 3 nước có điểm heuristic cao nhất
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
