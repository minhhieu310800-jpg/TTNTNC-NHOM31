# Bộ nhớ đỉnh theo tác tử (mode_a_0.5s)

Máy: macOS-14.6.1-arm64-arm-64bit-Mach-O, Python 3.15.2. Mỗi tác tử chạy trong tiến trình riêng, 1 luồng PyTorch; RSS nền đã gồm Python, NumPy, PyTorch và engine.

| tác tử | loại | số ván | số nước | RSS nền (MB) | RSS đỉnh (MB) | tăng do tác tử (MB) |
|---|---|---|---|---|---|---|
| random | random | 6 | 20 | 244.7 | 244.8 | 0.1 |
| heuristic | heuristic | 6 | 354 | 245.2 | 245.5 | 0.2 |
| alphabeta | alphabeta | 6 | 125 | 244.9 | 277.4 | 32.5 |
| mcts | mcts | 6 | 28 | 244.1 | 294.0 | 49.9 |
| sl | sl | 6 | 106 | 244.2 | 250.6 | 6.3 |
| rl | rl | 6 | 118 | 245.0 | 251.6 | 6.6 |
| puct | puct | 6 | 169 | 244.5 | 255.3 | 10.8 |
| alphazero | puct | 6 | 89 | 244.2 | 256.1 | 11.9 |
