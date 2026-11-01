# Lab05 – Tìm kiếm đối kháng và trò chơi

Bài lab môn Trí tuệ nhân tạo nâng cao (TTNTNC) – Nhóm 31.

Nhóm làm Bài 1–9 theo đề, bỏ Bài 10 (Dots and Boxes). Toàn bộ code dùng **đúng code gốc của thầy**, không thêm file hay hàm phụ.

> Thư mục này tách riêng với đồ án Caro (`do-an-caro/`). Mọi thay đổi của Lab05 chỉ nằm trong `lab05-tim-kiem-doi-khang/`.

## Phân công

| Thành viên | GitHub | Bài (issue) | Notebook |
| --- | --- | --- | --- |
| Huỳnh Nhật Quang | @HuynhQuang16032006 | Bài 1 (#24), 2 (#25), 7 (#30) + gom bảng tổng hợp, README | `tictactoe_definitions`, `tictactoe_and_or_tree_search`, `tictactoe_interactive` |
| Lương Tùng Quang | @quang-2703 | Bài 3 (#26), 4 (#27) | `tictactoe_alpha_beta_tree_search`, `tictactoe_heuristic_alpha_beta_tree_search` |
| Phạm Đức Lương | @DucLuong2201 | Bài 5 (#28), 6 (#29) | `tictactoe_pure_monte_carlo_search`, `tictactoe_monte_carlo_tree_search_restricted` |
| Lê Minh Hiếu | @minhhieu310800-jpg | Bài 8 (#31), 9 (#32) | `assignment_connect4`, `assignment_mean_connect4` |

Mỗi bài có một issue trong milestone **Lab05 – Tìm kiếm đối kháng và trò chơi**, **hạn 09/11/2026**. Lưu ý chi tiết từng bài nằm trong comment của issue.

## Cấu trúc thư mục

```
lab05-tim-kiem-doi-khang/
├── tictactoe.py                                        # Hàm dùng chung của thầy: bàn cờ, play(), random_player
├── tictactoe_definitions.ipynb                         # Bài 1
├── tictactoe_and_or_tree_search.ipynb                  # Bài 2
├── tictactoe_alpha_beta_tree_search.ipynb              # Bài 3
├── tictactoe_heuristic_alpha_beta_tree_search.ipynb    # Bài 4
├── tictactoe_pure_monte_carlo_search.ipynb             # Bài 5
├── tictactoe_monte_carlo_tree_search_restricted.ipynb  # Bài 6
├── tictactoe_interactive.ipynb                         # Bài 7
├── assignment_connect4.ipynb                           # Bài 8
└── assignment_mean_connect4.ipynb                      # Bài 9
```

## Cách chạy

```bash
cd lab05-tim-kiem-doi-khang
pip install numpy pandas matplotlib jupyterlab
jupyter lab
```

Mở notebook của bài cần chạy, chọn **Kernel → Restart Kernel and Run All Cells**. Bài 7 phải chạy trong JupyterLab để nhập nước đi.

## Quy ước làm bài

1. **Bàn kiểm thử:** dùng đúng 4 bàn có sẵn trong notebook: X sắp thắng, X cần hòa, tình huống bất lợi, bàn rỗng.
2. **Số nút, thời gian trên bàn kiểm thử:** để `DEBUG = 1`, lấy số nút ở dòng `Number of nodes searched`, thời gian ở `Wall time` của `%time` / `%timeit`.
3. **Đấu với Random:** dùng `play()` (mặc định 100 ván), chạy cả 2 chiều như notebook, đặt `DEBUG = 0`. Riêng Bài 1 dùng `N = 1000`.
4. **Thời gian TB/nước đi** (bảng tổng hợp): trung bình `Wall time` của hàm tìm kiếm trên 4 bàn kiểm thử.
5. **Cấu hình ghi vào bảng tổng hợp:** Heuristic Alpha-Beta ghi 2 dòng (cutoff = 2 và 4), Pure Monte Carlo dùng `pmcs100_player`, UCB1 dùng `ucb1_100_player`.
6. **Nhận xét:** viết trong ô markdown ngay dưới mỗi thí nghiệm, mở đầu bằng **Nhận xét:**. Không cố định seed, ghi đúng số trong output của mình.

## Quy trình Git

1. Tạo nhánh từ `main`: `lab05/<số issue>-bai-<n>`, ví dụ `lab05/24-bai-1`.
2. **Chỉ sửa file trong `lab05-tim-kiem-doi-khang/`**, và chỉ sửa notebook của bài mình. Không đụng vào `do-an-caro/` hay notebook của người khác.
3. Trước khi commit: Restart & Run All để notebook có đủ output.
4. Mở Pull Request vào `main`, ghi `Closes #<số issue>` trong mô tả. Kiểm tra tab **Files changed** chỉ có file trong `lab05-tim-kiem-doi-khang/` rồi mới merge.

## Bảng tổng hợp (mục 4 của đề)

Huỳnh Nhật Quang gom số liệu từ các issue vào bảng này khi cả nhóm làm xong.

| Tác tử/Thuật toán | Đối thủ | Số ván | Thắng | Hòa | Thua | Thời gian TB/nước đi |
| --- | --- | --- | --- | --- | --- | --- |
| Random | Random | 1000 | | | | |
| AND-OR | Random | 100 | | | | |
| Alpha-Beta | Random | 100 | | | | |
| Heuristic Alpha-Beta (cutoff = 2) | Random | 100 | | | | |
| Heuristic Alpha-Beta (cutoff = 4) | Random | 100 | | | | |
| Pure Monte Carlo (N = 100) | Random | 100 | | | | |
| UCB1/MCTS (N = 100) | Random | 100 | | | | |
