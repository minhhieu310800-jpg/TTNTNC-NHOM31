# Arena: giải đấu round-robin và thống kê

Chạy các lệnh dưới đây trong thư mục `do-an-caro/`.

## 1. Khai cuộc

`configs/openings.json` chứa **100 khai cuộc cân bằng** dùng chung cho mọi trận:
2–4 nước ngẫu nhiên trong ô 5×5 quanh tâm (seed 2026), được Alpha-Beta độ sâu 2
chấm |điểm| ≤ 300 (không bên nào lợi rõ), không trùng nhau qua 8 phép đối xứng.
File đã commit sẵn, **không cần sinh lại**. Lệnh sinh lại (ra đúng file cũ):

```bash
python -m arena.openings --n 100 --seed 2026 --out configs/openings.json
```

## 2. Chạy giải đấu

```bash
python -m arena.tournament configs/arena/smoke.yaml      # chạy thử, vài chục giây
python -m arena.tournament configs/arena/mode_a.yaml     # chế độ A: 0,5 s/nước
python -m arena.tournament configs/arena/mode_b.yaml     # chế độ B: cấu hình mặc định
```

- Mỗi cặp: mỗi khai cuộc chơi 2 ván đổi bên đi trước → 100 khai cuộc = 200 ván/cặp.
- Cặp có tác tử ngẫu nhiên (`random`, `mcts`) chạy thêm theo mọi seed trong `seeds`.
- `workers`: số tiến trình song song. Khi đo thời gian (chế độ A), không đặt quá số nhân CPU vật lý.
- **Bị ngắt thì chạy lại đúng lệnh cũ**: các ván đã có trong log được bỏ qua.
- `--limit 20`: chỉ chạy 20 ván còn thiếu (thử nhanh).

Thêm tác tử mới (cấp 5–8): đăng ký lớp trong `arena/agent_factory.py` (`AGENT_TYPES`),
rồi khai báo trong mục `agents` của file YAML với `type` và `params`.

## 3. Log

`logs/<name>.jsonl`, mỗi ván một dòng:

| Trường | Nội dung |
|---|---|
| `game_id`, `experiment`, `mode`, `budget` | định danh ván và chế độ |
| `engine_version`, `rule`, `size` | phiên bản engine và luật |
| `opening_id`, `opening_moves`, `seed` | khai cuộc và seed lần chạy |
| `agent_x`, `agent_o` | nhãn, loại, tham số, seed của tác tử cầm X (đi trước) / O |
| `moves` | mỗi nước: `player`, `action`, `time_ms`, `stats` (lấy từ `last_stats`) |
| `result`, `winner` | +1 X thắng, −1 O thắng, 0 hòa; nhãn tác tử thắng |
| `error` | lỗi nếu tác tử ném ngoại lệ hoặc đi nước bất hợp lệ (bị xử thua) |

`logs/<name>.meta.jsonl`: mỗi lần chạy một dòng gồm cấu hình, commit git, cấu hình máy.

`logs/` nằm trong `.gitignore`. Log của giải đấu chính thức (#15) sẽ đưa lên Google Drive hoặc nộp kèm.

## 4. Thống kê và hình

```bash
python -m analysis.summarize logs/smoke.jsonl              # in bảng Markdown
python -m analysis.summarize logs/mode_a_0.5s.jsonl --out bang_mode_a.md
```

Gồm: ma trận đối đầu (score hàng gặp cột, Wilson 95%), thắng/hòa/thua từng cặp,
score theo màu quân (lợi thế đi trước), thời gian trung bình và p95 mỗi nước,
số nút / số mô phỏng trung bình, Elo Bradley-Terry kèm khoảng tin cậy bootstrap.

Notebook `analysis/arena_report.ipynb` vẽ các bảng và hình trên; đổi `LOG_PATH` rồi Run All,
hình lưu vào `analysis/figures/`.

## 5. Cấp 7 và RQ2 (issue #13)

Cấp 7 (AlphaGo-inspired) là `PUCTAgent` (`type: puct`) dùng **mạng cấp 5** đã huấn luyện:
policy của mạng làm xác suất prior P, value của mạng thay rollout khi đánh giá lá.
Mạng giữ nguyên, không có vòng lặp self-play cập nhật mạng như cấp 8.

```yaml
cap7:
  type: puct
  params: {checkpoint: checkpoints/level5/best.pt, simulations: 400, device: cpu}
```

`configs/arena/rq2.yaml` trả lời RQ2: ở cùng 400 / 800 / 1.600 mô phỏng, so PUCT + mạng cấp 5
với policy-only (cấp 5, argmax policy) và MCTS thuần (cấp 4). Mỗi mức 3 cặp, mỗi cặp 200 ván,
chế độ B (so theo số mô phỏng; thời gian mỗi nước vẫn ghi trong log).

```bash
# Cần checkpoints/level5/best.pt (tải từ Drive MyDrive/caro/level5/best.pt)
python -m arena.tournament configs/arena/rq2.yaml --limit 20   # chạy thử
python -m arena.tournament configs/arena/rq2.yaml              # chạy đủ 1.800 ván
python -m analysis.summarize logs/rq2_puct.jsonl --out bang_rq2.md
```

Giả thuyết trong proposal: PUCT + mạng > MCTS thuần > policy-only. Khi có mạng cấp 6 (#11),
bỏ chú thích `puct_rl_400` và thêm cặp `[puct_rl_400, puct_400]` vào `pairs` để so PUCT dùng mạng cấp 5 với mạng cấp 6.
