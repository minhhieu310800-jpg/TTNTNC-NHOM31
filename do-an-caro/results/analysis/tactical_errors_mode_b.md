# Lỗi chiến thuật theo tác tử (mode_b_default.jsonl.gz, 5600 ván)

Đếm trên mọi nước đi của tác tử. Bỏ lỡ thắng: có nước thắng ngay nhưng không đi. Không chặn: đối thủ có đúng một nước thắng ở lượt sau và sau nước đi của tác tử vẫn còn nước thắng (chặn hợp lệ gồm cả bịt đầu còn lại theo luật chặn 2 đầu), trong khi chặn thì chưa thua ngay. Chặn vô ích: không chặn nhưng dù chặn thì đối thủ vẫn tạo được hai đường thắng ở nước kế tiếp. Thế thua chắc: đối thủ đã có từ hai nước thắng. Hai loại sau không tính là lỗi.

| tác tử | số ván | số nước | bỏ lỡ thắng | không chặn | lỗi / 100 nước | % ván có lỗi | chặn vô ích | thế thua chắc |
|---|---|---|---|---|---|---|---|---|
| heuristic | 1400 | 19366 | 0 | 0 | 0.00 | 0.0 | 0 | 383 |
| alphabeta | 1400 | 13494 | 0 | 0 | 0.00 | 0.0 | 0 | 512 |
| sl | 1400 | 11614 | 26 | 20 | 0.40 | 3.1 | 10 | 553 |
| rl | 1400 | 12499 | 31 | 19 | 0.40 | 3.5 | 17 | 478 |
| puct | 1400 | 14034 | 29 | 33 | 0.44 | 4.3 | 8 | 54 |
| mcts | 1400 | 7485 | 3 | 89 | 1.23 | 6.0 | 37 | 1100 |
| random | 1400 | 6943 | 21 | 88 | 1.57 | 6.8 | 36 | 1275 |
| alphazero | 1400 | 11676 | 15 | 171 | 1.59 | 12.6 | 50 | 684 |
