# Đồ án Caro 11×11 – Game AI đa cấp độ

Đồ án cuối kỳ môn Trí tuệ nhân tạo nâng cao (TTNTNC), Trường Đại học Sài Gòn, Nhóm 31. GVHD: TS. Đỗ Như Tài.

Đề tài xây dựng game Caro 11×11 theo luật Việt Nam (thắng khi có từ 5 quân liên tiếp không bị chặn hai đầu, biên không tính là chặn, đủ 121 ô mà chưa ai thắng thì hòa) với 8 cấp tác tử AI, từ Random đến AlphaZero-inspired, rồi đánh giá chúng bằng giải đấu vòng tròn có kiểm soát ngân sách tính toán.

| Cấp | Tác tử | `type` trong cấu hình | Ý tưởng |
|---|---|---|---|
| 1 | Random | `random` | Chọn đều một ô trống, làm mốc thấp nhất |
| 2 | Heuristic | `heuristic` | Sáu luật ưu tiên: thắng ngay, chặn ngay, bốn mở, chặn bốn/ba mở, đòn kép, điểm mẫu hình |
| 3 | Minimax/Alpha-Beta | `alphabeta` | Negamax, cắt tỉa, sắp nước, bảng chuyển vị Zobrist, đào sâu lặp |
| 4 | MCTS-UCT | `mcts` | UCT c = √2, rollout ngẫu nhiên tới hết ván |
| 5 | Mạng policy-value (SL) | `sl` | ResNet 6 khối học bắt chước Alpha-Beta độ sâu 3 |
| 6 | Học tăng cường (RL) | `rl` | Actor-critic self-play từ mạng cấp 5, pool đối thủ, phạt KL |
| 7 | AlphaGo-inspired | `puct` + mạng cấp 5 | PUCT 400 mô phỏng, prior và value từ mạng cấp 5 |
| 8 | AlphaZero-inspired | `puct` + mạng cấp 8 | Vòng lặp self-play → train → gating, mạng 4 khối học từ đầu |

Cấp 7 và 8 chỉ là phiên bản lấy cảm hứng, không tái hiện đầy đủ AlphaGo/AlphaZero; báo cáo (Bảng 2.1) ghi rõ các thành phần đã và chưa hiện thực.

Mục lục: [Cài đặt](#1-cài-đặt) · [Checkpoint](#2-checkpoint) · [Demo](#3-chạy-demo) · [Kiểm thử](#4-kiểm-thử) · [Huấn luyện](#5-huấn-luyện) · [Đánh giá](#6-đánh-giá-giải-đấu-và-thí-nghiệm) · [Tạo lại bảng, hình](#7-tạo-lại-bảng-và-hình-của-báo-cáo) · [Cấu trúc thư mục](#8-cấu-trúc-thư-mục) · [Quy trình nhóm](#10-quy-trình-làm-việc-nhóm)

## 1. Cài đặt

Mọi lệnh trong README chạy trong thư mục `do-an-caro/`.

```bash
git clone https://github.com/minhhieu310800-jpg/TTNTNC-NHOM31.git
cd TTNTNC-NHOM31/do-an-caro
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Khuyến nghị **Python 3.11–3.12**. Nhóm đã chạy được trên Python 3.11, 3.13 và 3.15. Trên Windows với Python 3.14 trở lên, `pygame` chưa có bản cài sẵn; khi đó cài `pip install pygame-ce` thay cho `pygame`.

Thư viện chính: NumPy, Numba (biên dịch hàm lượng giá), PyTorch (mạng cấp 5–8), PyYAML, pandas và matplotlib (thống kê, vẽ hình), Pygame (giao diện), pytest (kiểm thử). Máy không có GPU vẫn chạy được demo và mọi giải đấu, vì khi thi đấu các mạng đều suy luận trên CPU.

## 2. Checkpoint

Checkpoint không nằm trong Git (`checkpoints/` và `*.pt` có trong `.gitignore`). Lấy ba file từ thư mục `caro/` trên Google Drive của nhóm hoặc từ tệp ZIP nộp bài, đặt đúng đường dẫn sau (giao diện và cấu hình giải đấu đọc cố định các đường dẫn này):

| Cấp | Đường dẫn trong repo | Thư mục gốc trên Drive |
|---|---|---|
| 5 (và cấp 7 dùng chung) | `checkpoints/level5/best.pt` | `MyDrive/caro/level5/best.pt` |
| 6 | `checkpoints/level6/best.pt` | `MyDrive/caro/level6_v2/best.pt` |
| 8 | `checkpoints/alphazero2/best.pt` | `MyDrive/caro/alphazero2/best.pt` |

Kiểm tra đúng bản dùng trong báo cáo bằng mã SHA-256 đã lưu ở `results/training/checkpoints_sha256.json`:

```bash
python -c "import hashlib,json;[print(k, hashlib.sha256(open(v['checkpoint'],'rb').read()).hexdigest()==v['sha256']) for k,v in json.load(open('results/training/checkpoints_sha256.json')).items()]"
```

Cả ba dòng phải in `True`. Thiếu checkpoint thì cấp tương ứng bị làm mờ trên giao diện; các cấp 1–4 vẫn chạy bình thường.

## 3. Chạy demo

```bash
python -m ui.app
```

- **Người – Máy:** chọn cầm X (đi trước) hoặc O, chọn cấp AI đối thủ, bấm vào ô để đi; có nút Hoàn tác.
- **Máy – Máy:** chọn cấp AI riêng cho X và O; có Tạm dừng / Tiếp tục và Đi 1 nước.
- **Thời gian suy nghĩ:** 0,1 / 0,5 / 1,0 giây mỗi nước.
- **AI cân nhắc:** khung bên phải hiện thông số của nước AI gần nhất (luật đã dùng, số nút, độ sâu, số mô phỏng, điểm, thời gian).
- **Xem lại ván:** các nút `|<  <  >  >|` hoặc phím ← → Home End, Space để tự phát.

Chi tiết ở [`ui/README.md`](ui/README.md).

## 4. Kiểm thử

```bash
pytest -q
```

Bộ kiểm thử có 260 test, bao phủ luật thắng/thua/hòa (bốn hướng, chuỗi dài hơn 5, chặn hai đầu, sát biên, bàn đầy), API engine, các tác tử, hệ thống giải đấu, thống kê, quy trình huấn luyện ở quy mô thu nhỏ và logic giao diện (chạy không cần cửa sổ). Chạy hết mất khoảng 1 phút.

## 5. Huấn luyện

Lần chạy chính thức: sinh dữ liệu chạy trên CPU, huấn luyện chạy trên GPU T4 của Google Colab. Mọi quy trình cố định seed, ghi cấu hình đã dùng, phiên bản engine, cấu hình máy và tự chạy tiếp khi bị ngắt (chạy lại đúng lệnh cũ). Log của lần chạy chính thức nằm trong `results/training/`.

| Bước | Lệnh chạy thử nhanh | Lần chạy chính thức | Thời gian |
|---|---|---|---|
| Sinh dữ liệu cấp 5 | `python -m training.generate_data configs/training/smoke.yaml` | 20.000 ván Alpha-Beta, CPU 12 luồng, 4 tiến trình | 9.123 s |
| Huấn luyện cấp 5 | `python -m training.train_supervised_policy_value_network configs/training/supervised-policy-value-network-smoke.yaml` | Tối đa 60 epoch, dừng sớm ở epoch 59, tốt nhất epoch 54 | 1.203 s (T4) |
| Huấn luyện cấp 6 | `python -m training.train_rl configs/training/rl_smoke.yaml` | 20.000 ván self-play, 20 vòng, tốt nhất vòng 15 | 151 s (T4) |
| Vòng lặp cấp 8 | `python -m training.alphazero configs/alphazero_smoke.yaml` | 25 vòng × 200 ván, 25/25 vòng qua gating | 12.144 s (T4) |

### Cấp 5 – sinh dữ liệu và học có giám sát

```bash
# Sinh 20.000 ván Alpha-Beta độ sâu 3 -> data/dataset_sl.npz (chia 70/10/20 theo ván, loại trùng qua 8 phép đối xứng)
python -m training.generate_data configs/training/generate_data.yaml --workers 4

# Huấn luyện (lần chính thức dùng --epochs 60, dừng sớm sau 5 epoch không cải thiện)
python -m training.train_supervised_policy_value_network configs/training/supervised-policy-value-network.yaml \
    --data data/dataset_sl.npz --epochs 60 --device auto
```

Kết quả: `checkpoints/level5/` (`best.pt`, `latest.pt`, `train_log.jsonl`, `summary.json`). Thống kê bộ dữ liệu: `python -m analysis.dataset_stats data/dataset_sl.npz`.

### Cấp 6 – học tăng cường qua self-play

```bash
python -m training.train_rl configs/training/rl.yaml --device auto
```

`configs/training/rl.yaml` là đúng cấu hình chính thức: khởi tạo từ `checkpoints/level5/best.pt`, lr 1e-5, τ = 1,0 cho cả hai bên, chuẩn hóa lợi thế, phạt 0,05·KL(π‖π_SL), mỗi vòng 1.000 ván rồi đấu 100 ván với cấp 5 để chọn `best.pt`. Kết quả ở `checkpoints/level6/`.

### Cấp 8 – vòng lặp AlphaZero-inspired

```bash
python -m training.alphazero configs/alphazero.yaml --rounds 25 --device auto
```

Mỗi vòng: self-play 200 ván bằng PUCT 100 mô phỏng (Dirichlet α = 0,15, ε = 0,25), huấn luyện 5 epoch trên bộ đệm 4 vòng gần nhất, gating 40 ván và chỉ nhận mạng mới khi đạt từ 0,55. Thêm `--max-hours 3` để tự dừng an toàn; chạy lại cùng lệnh sẽ tiếp tục từ `state.json`. Chi tiết ở [`training/ALPHAZERO.md`](training/ALPHAZERO.md).

### Chạy trên Google Colab

Mở notebook tương ứng, chọn **Runtime → Change runtime type → T4 GPU** rồi chạy lần lượt các ô; checkpoint được lưu về `MyDrive/caro/` nên phiên bị ngắt vẫn chạy tiếp được.

| Cấp | Notebook |
|---|---|
| 5 | [`training/supervised-policy-value-network-colab.ipynb`](training/supervised-policy-value-network-colab.ipynb) (cần tải `dataset_sl.npz` lên `MyDrive/caro/data/`) |
| 6 | [`training/rl_colab.ipynb`](training/rl_colab.ipynb) (cần `MyDrive/caro/level5/best.pt`) |
| 8 | [`training/alphazero_colab.ipynb`](training/alphazero_colab.ipynb) |

Hướng dẫn chi tiết từng bước: [`training/README.md`](training/README.md).

## 6. Đánh giá: giải đấu và thí nghiệm

Mọi giải đấu dùng chung `configs/openings.json`: 100 khai cuộc cân bằng (seed 2026). Mỗi cặp chơi 2 ván trên mỗi khai cuộc, đổi bên đi trước, tức 200 ván mỗi cặp; cặp có tác tử ngẫu nhiên (Random, MCTS) chạy thêm theo các seed trong cấu hình. Bị ngắt thì chạy lại đúng lệnh, các ván đã có trong log được bỏ qua.

```bash
python -m arena.tournament configs/arena/smoke.yaml          # chạy thử vài chục giây
python -m arena.tournament configs/arena/mode_a.yaml         # thêm --limit 20 để chỉ chạy 20 ván
```

| Thí nghiệm | Lệnh | Số ván | Log chính thức |
|---|---|---|---|
| Giải đấu chế độ A (0,5 s/nước) | `python -m arena.tournament configs/arena/mode_a.yaml` | 10.800 | `results/issue15/mode_a_0.5s.*` |
| Giải đấu chế độ B (cấu hình mặc định) | `python -m arena.tournament configs/arena/mode_b.yaml` | 10.800 | `results/issue15/mode_b_default.*` |
| RQ1 và ablation Alpha-Beta, MCTS | `python -m analysis.ablation --workers 8` | 4.800 | `results/issue8/` |
| RQ2: PUCT, policy-only, MCTS | `python -m arena.tournament configs/arena/rq2.yaml` | 4.200 | `results/issue15/rq2_puct.*` |
| RQ3: cấp 6 gặp cấp 5 và Heuristic | `python -m arena.tournament configs/arena/rq3.yaml` | 2.200 | `results/issue11/rq3_rl_vs_sl.*` |
| Độ sâu Alpha-Beta 2, 3, 4 | `python -m arena.tournament configs/arena/ab_depth.yaml` | 2.400 | `results/issue15/ab_depth.*` |
| Đối đầu checkpoint cấp 8 | `python -m arena.tournament configs/arena/az_ladder.yaml` | 4.800 | `results/issue14/az_ladder.*` |

Lưu ý khi chạy lại:

- Chế độ A đo theo thời gian, nên chạy trên **một máy**, số tiến trình (`workers`) không vượt quá số nhân vật lý. Chế độ B so theo cấu hình (độ sâu, số mô phỏng) nên có thể chia cặp cho nhiều máy rồi gộp log.
- Ngoài ba checkpoint ở mục 2, `az_ladder.yaml` cần thêm mạng cấp 8 ở các vòng 1, 5, 10, 15, 20, 25 (`checkpoints/alphazero2/rounds/round_0NN.pt`), còn `rq3.yaml` cần mạng cấp 6 ở các vòng 5, 10, 15, 20 (`checkpoints/level6/rounds/round_0NN.pt`). Các file này cũng nằm trên Drive, trong thư mục `rounds/` của từng cấp.
- Log mới ghi vào `logs/<name>.jsonl` (mỗi ván một dòng: engine, luật, seed, khai cuộc, hai tác tử, từng nước kèm thời gian và `last_stats`, kết quả) và `logs/<name>.meta.jsonl` (cấu hình, commit Git, cấu hình máy). Định dạng chi tiết ở [`arena/README.md`](arena/README.md).

Tổng hợp một log thành bảng Markdown (ma trận đối đầu kèm Wilson 95%, thắng/hòa/thua, score theo màu quân, thời gian trung bình và p95, số nút hoặc mô phỏng, Elo Bradley-Terry kèm bootstrap):

```bash
python -m analysis.summarize logs/mode_a_0.5s.jsonl --out bang_mode_a.md

# Log chính thức được nén .gz; giải nén trước (giữ nguyên file .gz):
python -m gzip -d results/issue15/rq2_puct.jsonl.gz
python -m analysis.summarize results/issue15/rq2_puct.jsonl
```

## 7. Tạo lại bảng và hình của báo cáo

Toàn bộ bảng và hình ở Chương 4 được sinh từ log đã commit trong `results/`, không cần chạy lại thí nghiệm:

```bash
python -m analysis.report_figures            # mọi hình -> analysis/figures/ (--only heatmap để vẽ một nhóm)
python -m analysis.report_tables             # môi trường chạy, độ phức tạp -> results/analysis/
python -m analysis.game_analysis errors results/issue15/mode_b_default.jsonl.gz --seed 0 \
    --out results/analysis/tactical_errors_mode_b.md   # lỗi chiến thuật một nước (Bảng 4.13)
python -m analysis.game_analysis boards      # hình minh họa ván điển hình (Hình 4.16–4.19)
```

Đo bộ nhớ (Bảng 4.6): `python -m analysis.memory_probe configs/arena/mode_a.yaml --out results/memory/mode_a.json` khi thi đấu, thêm `--train` để đo RAM một bước huấn luyện. VRAM trên GPU T4 đo bằng notebook [`analysis/do_vram_t4.ipynb`](analysis/do_vram_t4.ipynb) chạy trên Colab. Notebook [`analysis/arena_report.ipynb`](analysis/arena_report.ipynb) vẽ nhanh bảng và hình cho một log bất kỳ (đổi `LOG_PATH` rồi Run All).

## 8. Cấu trúc thư mục

```
do-an-caro/
├── game/        # Engine CaroGame: luật Việt Nam, API reset/step/legal_actions/undo/clone/render, nhận diện mẫu hình
├── agents/      # 8 cấp tác tử, mạng PolicyValueNet dùng chung, PUCT
├── arena/       # Khai cuộc cân bằng, lập lịch, chạy giải đấu đa tiến trình, ghi log JSONL
├── training/    # Sinh dữ liệu, huấn luyện cấp 5, 6, 8 và notebook Colab
├── analysis/    # Thống kê (Wilson, Bradley-Terry, bootstrap), script sinh bảng/hình, hình trong figures/
├── ui/          # Giao diện Pygame
├── configs/     # YAML: arena/ (giải đấu, RQ), training/ (huấn luyện), openings.json
├── results/     # Log thi đấu nén, log huấn luyện, kết quả đo bộ nhớ và bảng tổng hợp
├── tests/       # pytest
└── requirements.txt
```

| Thư mục trong `results/` | Nội dung |
|---|---|
| `issue8/` | RQ1 (0,1 / 0,5 / 1,0 s), ablation số mô phỏng và rollout MCTS, ablation cắt tỉa Alpha-Beta |
| `issue11/` | RQ3 (chạy lại, khớp từng ván với lần chạy gốc) |
| `issue14/` | Đối đầu checkpoint cấp 8 với Heuristic và cấp 5 |
| `issue15/` | Giải đấu chế độ A, B; RQ2; độ sâu Alpha-Beta; phân tích seed |
| `training/` | Log huấn luyện, cấu hình đã dùng và thống kê dữ liệu cấp 5, 6, 8; SHA-256 checkpoint |
| `memory/` | RAM khi thi đấu (Windows, Apple M1) và VRAM T4 khi huấn luyện |
| `analysis/` | Lỗi chiến thuật, ván điển hình, môi trường chạy, độ phức tạp |

## 9. Thành viên

| Thành viên | MSSV | GitHub | Phụ trách chính |
|---|---|---|---|
| Lê Minh Hiếu (nhóm trưởng) | 3124720023 | @minhhieu310800-jpg | Arena và thống kê, cấp 7–8, giải đấu chính thức, RQ2, phân tích số liệu, báo cáo |
| Phạm Đức Lương | 3124720042 | @DucLuong2201 | Sinh dữ liệu, cấp 5–6, RQ3 |
| Lương Tùng Quang | 3124720057 | @quang-2703 | Cấp 3–4, ablation Alpha-Beta và MCTS, RQ1, chạy giải đấu chế độ A |
| Huỳnh Nhật Quang | 3124720056 | @HuynhQuang16032006 | Engine, kiểm thử luật, cấp 1–2, giao diện Pygame, chạy giải đấu chế độ B |

Bảng phân công và tỷ lệ đóng góp đầy đủ nằm ở cuối báo cáo.

## 10. Quy trình làm việc nhóm

1. **Mỗi việc là một issue.** Không code việc gì chưa có issue.
2. **Tạo nhánh từ `main` mới nhất:**
   ```bash
   git switch main
   git pull
   git switch -c caro/<số-issue>-<tên-ngắn>
   ```
3. **Code, commit, push nhánh của mình:** `git push -u origin caro/<số-issue>-<tên-ngắn>`
4. **Mở pull request vào `main`**, phần mô tả ghi `Closes #<số-issue>`.
5. **Ít nhất 1 người khác review** rồi mới merge. Merge xong thì xóa nhánh. **Không push thẳng lên `main`.**

Tên nhánh: `caro/<số issue>-<tên ngắn>`, viết thường, không dấu, nối bằng gạch ngang (ví dụ `caro/6-alphabeta`). Commit message ngắn, bắt đầu bằng loại thay đổi (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`), mỗi commit một ý; lịch sử commit được dùng để tính tỷ lệ đóng góp.

Trước khi mở pull request: đã `git pull` `main` mới nhất và xử lý xung đột, `pytest` pass, không commit file thừa, hàm chính có docstring ngắn.

### Quy ước code

- Tuân theo PEP 8; tên biến, hàm bằng tiếng Anh; có type hint cho hàm công khai.
- Không dùng số "ma thuật": kích thước bàn, độ sâu, số mô phỏng, seed đặt trong `configs/` hoặc hằng số đầu file.
- Mọi thứ có ngẫu nhiên phải nhận `seed` để chạy lại ra đúng kết quả.
- Mọi tác tử kế thừa `agents/base.py` và cài `select_action(state, budget=None) -> int` (chỉ số ô 0–120 = hàng × 11 + cột); không sửa `state` truyền vào (dùng `clone()` hoặc `step()` + `undo()`), ghi thông số suy nghĩ vào `last_stats`.
- Engine được khóa từ ngày 8/11 (tag `engine-v1.0`, phiên bản 1.0.0): chỉ sửa lỗi, không đổi luật hay API, để mọi giải đấu dùng cùng một phiên bản.

### Không đưa lên Git

Môi trường ảo (`.venv/`, `__pycache__/`), dữ liệu lớn, checkpoint, log thô (`data/`, `checkpoints/`, `logs/`, `*.pt`, `*.pth`, `*.npz`) và file riêng của máy (`.vscode/`, `.idea/`, `.DS_Store`) đã có trong `.gitignore`. Checkpoint và dữ liệu lớn để trên Google Drive chung của nhóm; log thi đấu chính thức sau khi kiểm tra được nén `.gz` và lưu trong `results/`.

## 11. Nguồn tham khảo và liêm chính học thuật

Lõi game, tám cấp tác tử, hệ thống giải đấu, quy trình huấn luyện, thống kê và giao diện do nhóm tự viết; không dùng engine Gomoku hay mã tác tử có sẵn. Thư viện nền: NumPy, Numba, PyTorch, pandas, matplotlib, Pygame, pytest. Ý tưởng thuật toán theo Russell & Norvig (2021), Kocsis & Szepesvári (2006), Silver và cộng sự (2016, 2017, 2018), Leela Zero (kỹ thuật FPU), Mark Liu, *AlphaGo Simplified* (2024); danh mục đầy đủ ở phần Tài liệu tham khảo của báo cáo. Nhóm có dùng công cụ AI (Claude Code, OpenAI Codex) hỗ trợ viết mã, test và rà soát; mọi đoạn mã đều được thành viên phụ trách đọc lại, chạy kiểm thử và được người khác duyệt qua pull request. Mọi số liệu đều sinh từ việc chạy thật mã nguồn và kiểm chứng được từ log trong `results/`.
