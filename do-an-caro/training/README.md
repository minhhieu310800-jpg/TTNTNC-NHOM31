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

## 3. Huấn luyện Supervised Policy-Value Network – Cấp 5 (Issue #10)

### Nguyên lý thiết kế (Proposal mục 4–5):
- **Kiến trúc:** `PolicyValueNet(blocks=6, channels=64, size=11)` hai đầu:
  - Đầu Policy: 121 logits, mask $-\infty$ ô đã có quân rồi softmax.
  - Đầu Value: $v \in [-1, 1]$ qua tanh, ước lượng kết quả theo góc nhìn người đến lượt.
- **Hàm mất mát:** $\text{Loss} = \text{Cross-Entropy}(\text{policy}) + \text{MSE}(\text{value}) + L_2\text{-regularization}$.
- **Tối ưu:** Adam, lr $10^{-3}$, batch size 256, tăng cường dữ liệu bằng 8 phép đối xứng ngẫu nhiên khi train.
- **Early Stopping & Checkpoint:** Theo dõi `val_loss`, tự động dừng sớm nếu không cải thiện sau `patience` epochs; lưu checkpoint `best.pt` và `latest.pt`.
- **Tác tử Cấp 5:** `SupervisedPolicyValueNetworkAgent` (alias `SLAgent`, `PolicyAgent` trong [`agents/supervised_policy_value_network_agent.py`](../agents/supervised_policy_value_network_agent.py)), chơi bằng **argmax policy** (chọn ô có xác suất cao nhất trong các ô hợp lệ).

### Lệnh chạy huấn luyện

Chạy thử nghiệm nhanh (Smoke test):
```bash
python -m training.train_supervised_policy_value_network configs/training/supervised-policy-value-network-smoke.yaml
```

Chạy huấn luyện chính thức:
```bash
# Huấn luyện trên GPU (nếu có) hoặc CPU
python -m training.train_supervised_policy_value_network configs/training/supervised-policy-value-network.yaml --data data/dataset_sl.npz --device auto

# Chạy tiếp từ checkpoint nếu bị ngắt:
python -m training.train_supervised_policy_value_network configs/training/supervised-policy-value-network.yaml --resume checkpoints/supervised-policy-value-network/latest.pt
```

### Chạy trên Google Colab GPU T4
Mở notebook [`training/supervised-policy-value-network-colab.ipynb`](supervised-policy-value-network-colab.ipynb):
1. Vào **Runtime → Change runtime type → T4 GPU**.
2. Chạy lần lượt các ô: tự động mount Drive, kéo mã nguồn từ GitHub, tải/sinh dữ liệu, train bằng GPU T4, vẽ biểu đồ Loss/Accuracy và chạy thử nghiệm tác tử Cấp 5.
3. Checkpoint được lưu tự động về Google Drive tại `/content/drive/MyDrive/caro/supervised-policy-value-network`.

### Sử dụng Tác tử Cấp 5 trong Arena & Mã nguồn

```python
from agents import SupervisedPolicyValueNetworkAgent
from game.game import CaroGame

# Khởi tạo tác tử với checkpoint
agent = SupervisedPolicyValueNetworkAgent(checkpoint="checkpoints/supervised-policy-value-network/best.pt", device="auto")

# Chọn nước đi (argmax policy)
game = CaroGame()
action = agent.select_action(game)
```

Cấu hình trong giải đấu Arena (`configs/arena/*.yaml`):
```yaml
agents:
  spvn:
    type: supervised_policy_value_network
    params:
      checkpoint: checkpoints/supervised-policy-value-network/best.pt
      device: auto
```

