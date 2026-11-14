# Môi trường chạy

| công việc | máy | thiết bị tính | số worker | số ván / thời gian | Python |
|---|---|---|---|---|---|
| RQ1 + ablation MCTS/rollout (#8) | Windows, Intel x86-64 (16 luồng) | CPU | 8 | 4.800 | 3.11.9 |
| Giải chính thức chế độ A (0,5 s/nước) | Windows, Intel x86-64 (16 luồng) | CPU | 4 | 10.800 | 3.11.9 |
| Giải chính thức chế độ B (mặc định) | Windows, Intel x86-64 (22 luồng) | CPU | 20 | 10.800 | 3.11.9 |
| RQ2: PUCT / policy / MCTS | Windows, Intel x86-64 (22 luồng) | CPU | 20 | 4.200 | 3.11.9 |
| Độ sâu Alpha-Beta 2/3/4 | Windows, Intel x86-64 (22 luồng) | CPU | 20 | 2.400 | 3.11.9 |
| RQ3: RL gặp SL và Heuristic (chạy lại) | macOS, Apple M1 (8 nhân) | CPU | 6 | 2.200 | 3.15.2 |
| Đối đầu checkpoint cấp 8 | macOS, Apple M1 (8 nhân) | CPU | 7 | 4.800 | 3.15.2 |
| Sinh dữ liệu cấp 5 (20.000 ván Alpha-Beta) | Windows 11, Intel x86-64 (12 luồng) | CPU | 4 | 9.123 s | 3.13.9 |
| Huấn luyện cấp 5 (SL) | Google Colab | GPU NVIDIA T4 (*) | – | 1.203 s | – |
| Huấn luyện cấp 6 (RL self-play) | Google Colab | GPU NVIDIA T4 (*) | – | 151 s | – |
| Huấn luyện cấp 8 (25 vòng) | Google Colab (2 CPU, 12.7 GB RAM) | GPU NVIDIA T4 (*) | 1 | 12.144 s (3.37 giờ) | 3.13.16 |
| Đo RAM từng tác tử | macOS, Apple M1 (8 nhân, 8 GB) | CPU | 1 | – | 3.15.2 |

(*) Log huấn luyện ghi `device: cuda` và thư mục `/content/drive/...` (Google Colab) nhưng không ghi tên GPU; loại GPU được xác nhận bằng phép đo VRAM trên cùng loại runtime Colab: Tesla T4, CUDA 13.0 (results/memory/train_vram_t4.json). Mọi tác tử dùng mạng nơ-ron khi thi đấu chạy trên CPU (`device: cpu`). Engine 1.0.0, NumPy 2.4.6 trên mọi máy chạy giải đấu.
