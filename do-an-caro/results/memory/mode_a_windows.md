# Bộ nhớ đỉnh theo tác tử (mode_a_0.5s)

Máy: Windows-10-10.0.26200-SP0, Python 3.11.9. Mỗi tác tử chạy trong tiến trình riêng, 1 luồng PyTorch; RSS nền đã gồm Python, NumPy, PyTorch và engine.

| tác tử | loại | số ván | số nước | RSS nền (MB) | RSS đỉnh (MB) | tăng do tác tử (MB) |
|---|---|---|---|---|---|---|
| random | random | 6 | 20 | 218.1 | 218.3 | 0.2 |
| heuristic | heuristic | 6 | 354 | 217.9 | 218.1 | 0.1 |
| alphabeta | alphabeta | 6 | 229 | 218.2 | 251.0 | 32.8 |
| mcts | mcts | 6 | 22 | 218.2 | 251.3 | 33.1 |
| sl | sl | 6 | 106 | 217.9 | 226.5 | 8.7 |
| rl | rl | 6 | 118 | 218.1 | 226.8 | 8.7 |
| puct | puct | 6 | 83 | 218.3 | 227.4 | 9.0 |
| alphazero | puct | 6 | 73 | 218.6 | 229.1 | 10.6 |
