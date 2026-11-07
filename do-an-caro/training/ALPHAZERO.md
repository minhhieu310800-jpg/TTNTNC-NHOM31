# Cấp 8 – AlphaZero-inspired

Chạy các lệnh trong thư mục `do-an-caro/`.

## Thành phần

| File | Vai trò |
|---|---|
| `agents/network.py` | Mạng policy-value dùng chung cho cấp 5, 7, 8 (input 4 mặt phẳng `get_planes()`, output policy 121 + value) và định dạng checkpoint |
| `agents/puct_agent.py` | Tìm kiếm PUCT (dùng chung cấp 7 và 8) và tác tử `PUCTAgent` cho arena |
| `training/selfplay.py` | Self-play nhiều ván cùng lúc, gom lá thành batch cho mạng; 8 phép đối xứng |
| `training/alphazero.py` | Vòng lặp self-play → train → gating, lưu trạng thái để chạy tiếp |
| `training/alphazero_colab.ipynb` | Chạy trên Google Colab, lưu mọi thứ lên Google Drive |
| `configs/alphazero.yaml` | Cấu hình chính (6 vòng × 100 ván, 100 mô phỏng/nước, mạng 4 khối × 64 kênh) |

Mạng **học từ đầu** (khởi tạo ngẫu nhiên), đúng proposal; không dùng mạng cấp 5.

## Chạy

```bash
# Thử nhanh (~30 giây) để kiểm tra máy
python -m training.alphazero configs/alphazero_smoke.yaml

# Chạy thật trên Mac/PC (CPU nhiều nhân): ~10 phút/vòng với 6 tiến trình (đo trên Mac 8 nhân)
python -m training.alphazero configs/alphazero.yaml --workers 6

# Thêm vòng sau khi đã xong 6 vòng
python -m training.alphazero configs/alphazero.yaml --workers 6 --rounds 10
```

Trên Colab: mở `training/alphazero_colab.ipynb`, chọn GPU T4, chạy lần lượt các ô.

**Bị ngắt thì chạy lại đúng lệnh cũ**: chương trình đọc `state.json` và làm tiếp từ
nhóm ván, bước train hoặc gating đang dở. `--max-hours 3` tự dừng an toàn sau 3 giờ.

## Kết quả trong `out_dir` (mặc định `checkpoints/alphazero/`, nằm trong `.gitignore`)

| File | Nội dung |
|---|---|
| `best.pt` | Mạng tốt nhất (đã qua gating), dùng cho self-play và thi đấu |
| `current.pt` | Mạng đang train (kèm trạng thái optimizer) |
| `rounds/round_XXX.pt` | Mạng sau mỗi vòng, để vẽ đường cong học tập bằng arena |
| `buffer/rXXX_gYYYY_NNN.npz` | Dữ liệu self-play (planes, pi, z, kết quả, độ dài ván) |
| `train_log.jsonl` | Mỗi vòng: loss, score gating + Wilson 95%, nhận/không, thống kê self-play, thời gian |
| `meta.jsonl` | Cấu hình, cấu hình máy, phiên bản engine và torch |

## Đưa vào giải đấu

```yaml
agents:
  az:
    type: puct
    params: {checkpoint: checkpoints/alphazero/best.pt, simulations: 400}
```

Cấp 7 dùng cùng loại `puct`, chỉ đổi `checkpoint` sang mạng cấp 5/6 (cùng định dạng `save_checkpoint`).

## Khác so với AlphaZero gốc (ghi trong báo cáo)

- Quy mô nhỏ hơn nhiều bậc: 6 vòng × 100 ván, 100 mô phỏng/nước, mạng 4 khối × 64 kênh
  (bản gốc: hàng triệu ván, 800 mô phỏng, 20–40 khối × 256 kênh).
- Input 4 mặt phẳng, chỉ có nước vừa đi (bản gốc dùng lịch sử nhiều nước).
- Có gating ≥ 55% kiểu AlphaGo Zero, trên 40 ván và khai cuộc cân bằng.
- Chạy trên CPU Mac hoặc Colab T4, không dùng TPU.
