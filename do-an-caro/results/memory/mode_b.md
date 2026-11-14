# Bộ nhớ đỉnh theo tác tử (mode_b_default)

Máy: macOS-14.6.1-arm64-arm-64bit-Mach-O, Python 3.15.2. Mỗi tác tử chạy trong tiến trình riêng, 1 luồng PyTorch; RSS nền đã gồm Python, NumPy, PyTorch và engine.

| tác tử | loại | số ván | số nước | RSS nền (MB) | RSS đỉnh (MB) | tăng do tác tử (MB) |
|---|---|---|---|---|---|---|
| random | random | 6 | 20 | 244.7 | 244.8 | 0.1 |
| heuristic | heuristic | 6 | 354 | 245.2 | 245.2 | 0.1 |
| alphabeta | alphabeta | 6 | 133 | 244.3 | 275.8 | 31.5 |
| mcts | mcts | 6 | 23 | 244.0 | 288.1 | 44.1 |
| sl | sl | 6 | 106 | 244.5 | 252.5 | 8.0 |
| rl | rl | 6 | 118 | 244.0 | 250.2 | 6.2 |
| puct | puct | 6 | 188 | 244.5 | 254.9 | 10.5 |
| alphazero | puct | 6 | 91 | 244.0 | 254.1 | 10.1 |
