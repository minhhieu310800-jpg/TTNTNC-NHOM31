# Ảnh hưởng của seed (giải đấu #15)

Score của A theo từng seed, chỉ các cặp có tác tử ngẫu nhiên (Random, MCTS); mỗi seed 200 ván (100 khai cuộc × 2 bên). ĐLC = độ lệch chuẩn mẫu giữa 3 seed.

## mode_a_0.5s

| A | B | seed 0 | seed 1 | seed 2 | TB ± ĐLC |
|---|---|---|---|---|---|
| random | heuristic | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | alphabeta | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | mcts | 0.000 | 0.000 | 0.010 | 0.003 ± 0.006 |
| random | sl | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | rl | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | puct | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | alphazero | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| heuristic | mcts | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| alphabeta | mcts | 0.995 | 0.995 | 1.000 | 0.997 ± 0.003 |
| mcts | sl | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| mcts | rl | 0.000 | 0.005 | 0.000 | 0.002 ± 0.003 |
| mcts | puct | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| mcts | alphazero | 0.005 | 0.010 | 0.005 | 0.007 ± 0.003 |

## mode_b_default

| A | B | seed 0 | seed 1 | seed 2 | TB ± ĐLC |
|---|---|---|---|---|---|
| random | heuristic | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | alphabeta | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | mcts | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | sl | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | rl | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | puct | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| random | alphazero | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| heuristic | mcts | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| alphabeta | mcts | 0.990 | 0.995 | 0.995 | 0.993 ± 0.003 |
| mcts | sl | 0.010 | 0.000 | 0.005 | 0.005 ± 0.005 |
| mcts | rl | 0.010 | 0.000 | 0.005 | 0.005 ± 0.005 |
| mcts | puct | 0.000 | 0.000 | 0.010 | 0.003 ± 0.006 |
| mcts | alphazero | 0.015 | 0.020 | 0.020 | 0.018 ± 0.003 |
