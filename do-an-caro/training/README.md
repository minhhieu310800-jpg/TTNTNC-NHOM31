# Training: Sinh dữ liệu huấn luyện và Supervised Learning (Issue #9, #10)

Module phục vụ sinh dữ liệu tự chơi từ Alpha-Beta, tiền xử lý, loại trùng theo đối xứng và huấn luyện mạng Policy-Value (Cấp 5) theo **Proposal mục 5**.

Chạy các lệnh dưới đây trong thư mục `do-an-caro/`.

## 1. Sinh dữ liệu từ Alpha-Beta (Issue #9)

### Nguyên lý thiết kế
- **Nguồn:** Khoảng 20.000 ván cờ tự chơi giữa hai tác tử Alpha-Beta (độ sâu 3, có nhiễu $\epsilon = 0.1$ để khám phá đa dạng thế cờ).
- **Khai cuộc:** 2–4 nước ngẫu nhiên trong vùng 5×5 quanh tâm để tạo các thế trận mở rộng.
- **Nhãn:**
  - `policy`: nước đi Alpha-Beta lựa chọn (chỉ số ô $0..120$).
  - `value`: kết quả ván cờ theo góc nhìn của người đến lượt ($+1.0$ chắc thắng, $-1.0$ chắc thua, $0.0$ hoà).
- **Phân chia không rò rỉ:** Chia tập Train/Validation/Test theo tỷ lệ $70 / 10 / 20$ **theo ván** (không chia theo từng thế cờ) để đảm bảo không rò rỉ các nước đi liên tiếp của cùng một ván vào tập kiểm thử.
- **Loại trùng theo 8 phép đối xứng và Trung bình Value:** Mỗi thế cờ được chuyển về dạng chuẩn tắc nhỏ nhất ($D_4$: 4 góc xoay $\times$ 2 lật) và băm để loại bỏ các bản trùng lặp. Khi các thế cờ trùng nhau, nhãn `value` được lấy trung bình (`average_values=True`) để phản ánh xác suất thắng/thua khách quan hơn.
- **Chống rò rỉ thế cờ giữa các tập (Anti-leakage):** Sau khi chuẩn hoá tập Train, các thế cờ trong Validation và Test có khóa chuẩn tắc trùng với Train sẽ tự động bị loại bỏ, ngăn hiện tượng rò rỉ thế cờ khai cuộc/tương đương vào tập kiểm thử.
- **Lưu Checkpoint theo chunk & Hỗ trợ tiếp tục chạy (Resume):** Tự động lưu mỗi chunk 500 ván thành file part (`data/parts/part_XXXX.npz`). Khi máy tắt hoặc bị gián đoạn, chỉ cần chạy lại cùng lệnh — hệ thống sẽ tự động quét các part đã xong và chỉ sinh tiếp các ván còn lại.
- **Tăng cường dữ liệu:** Khi huấn luyện (Cấp 5), lớp `CaroDataset` hỗ trợ tự động sinh ngẫu nhiên 8 phép đối xứng để tăng gấp 8 lần dữ liệu thực tế.

### Lệnh thực thi

Chạy thử nghiệm nhanh (Smoke test với RandomAgent, ~0.1 giây):
```bash
python -m training.generate_data configs/training/smoke.yaml
```

Chạy sinh dữ liệu chính thức với Alpha-Beta độ sâu 3:
```bash
# Chạy với 4 worker song song, lưu checkpoint mỗi 500 ván
python -m training.generate_data configs/training/generate_data.yaml --workers 4

# Chạy lại từ đầu bỏ qua part file cũ nếu muốn:
python -m training.generate_data configs/training/generate_data.yaml --no-resume

# Thử nghiệm với số lượng ván giới hạn (ví dụ 100 ván):
python -m training.generate_data configs/training/generate_data.yaml --limit 100 --workers 4
```

## 2. Cấu trúc dữ liệu và file lưu trữ

Dữ liệu được lưu trong thư mục `data/` (đã nằm trong `.gitignore`):

| File | Định dạng | Nội dung |
|---|---|---|
| `<parts_dir>/part_XXXX.npz` | NumPy nén | Checkpoint từng chunk (mỗi chunk 500 ván) để resume |
| `<name>.npz` | NumPy nén | Bộ dữ liệu hoàn chỉnh chứa mảng `train_*`, `val_*`, `test_*` (`planes`, `actions`, `values`, `game_ids`) |
| `<name>_train.npz` | NumPy nén | Riêng tập Train |
| `<name>_val.npz` | NumPy nén | Riêng tập Validation (đã lọc trùng với Train) |
| `<name>_test.npz` | NumPy nén | Riêng tập Test (đã lọc trùng với Train và Val) |
| `<name>_log.json` | JSON | Log đầy đủ: seed, cấu hình, phần cứng máy (`hardware_info`), số ván, thống kê trùng lặp/rò rỉ, thời gian sinh |

### Biểu diễn 4 mặt phẳng (`planes`):
1. **Mặt phẳng 0:** Quân của người chơi hiện tại ($1.0$ tại ô quân mình, $0.0$ nơi khác).
2. **Mặt phẳng 1:** Quân của đối thủ ($1.0$ tại ô đối thủ, $0.0$ nơi khác).
3. **Mặt phẳng 2:** Nước vừa đi cuối cùng ($1.0$ tại `last_move`, $0.0$ nơi khác).
4. **Mặt phẳng 3:** Lượt đi ($1.0$ nếu là Đen/X, $0.0$ nếu là Trắng/O).

## 3. Sử dụng `CaroDataset` trong huấn luyện (Issue #10)

```python
from training.dataset import CaroDataset

# Nạp tập train
train_dataset = CaroDataset.from_file("data/dataset_sl.npz", split="train", augment=True)

# Tương thích với PyTorch DataLoader khi có torch:
# loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
```
