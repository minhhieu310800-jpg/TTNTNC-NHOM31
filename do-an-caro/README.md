# Đồ án Caro 11x11 – Game AI đa cấp độ

Đồ án cuối kỳ môn Trí tuệ nhân tạo nâng cao (TTNTNC) – Trường Đại học Sài Gòn – Nhóm 31.

Xây dựng game Caro 11×11 (luật Việt Nam: thắng 5 quân, không bị chặn hai đầu) với 8 cấp tác tử AI, từ Random đến AlphaZero-inspired, và đánh giá bằng giải đấu round-robin.

> README này hiện chỉ ghi **quy ước làm việc chung**. Phần cài đặt, train và demo chi tiết sẽ bổ sung ở issue #17.

## Thành viên

| Thành viên | GitHub | Phụ trách |
| --- | --- | --- |
| Lê Minh Hiếu | @minhhieu310800-jpg | Cấp 7–8, arena và thống kê |
| Huỳnh Nhật Quang | @HuynhQuang16032006 | Engine, test, cấp 1–2, giao diện |
| Lương Tùng Quang | @quang-2703 | Cấp 3–4, ablation Alpha-Beta/MCTS |
| Phạm Đức Lương | @DucLuong2201 | Cấp 5–6, sinh dữ liệu |

## Cấu trúc thư mục

Mọi code của đồ án nằm trong `do-an-caro/`.

```
do-an-caro/
├── game/        # Engine: bàn cờ, luật, API (reset, step, legal_actions, ...)
├── agents/      # Tác tử cấp 1–8, mỗi cấp một file
├── arena/       # Cho các tác tử đấu nhau, ghi log
├── training/    # Sinh dữ liệu, train mạng, self-play
├── ui/          # Giao diện Pygame
├── analysis/    # Notebook vẽ bảng, biểu đồ từ log
├── tests/       # pytest
├── configs/     # File cấu hình YAML (seed, siêu tham số)
└── requirements.txt
```

## Cài đặt môi trường

```bash
cd do-an-caro
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dùng **Python 3.11**. Cài thêm thư viện mới thì thêm vào `requirements.txt` trong cùng pull request.

## Quy trình làm việc

1. **Mỗi việc là một issue.** Không code việc gì chưa có issue.
2. **Tạo nhánh từ `main` mới nhất.** Có thể tạo trong issue (Development → Create a branch, source: `main`) hoặc bằng lệnh:
   ```bash
   git switch main
   git pull
   git switch -c caro/<số-issue>-<tên-ngắn>
   ```
3. **Code, commit, push nhánh của mình:**
   ```bash
   git push -u origin caro/<số-issue>-<tên-ngắn>
   ```
4. **Mở pull request vào `main`.** Trong phần mô tả ghi `Closes #<số-issue>`.
5. **Ít nhất 1 người khác review** rồi mới merge. Merge xong thì xóa nhánh.

**Không push thẳng lên `main`.**

### Tên nhánh

`caro/<số issue>-<tên ngắn>`, viết thường, không dấu, nối bằng dấu gạch ngang.

```
caro/1-engine
caro/6-alphabeta
caro/10-sl-network
caro/12-arena
```

### Commit message

Viết tiếng Việt cũng được, ngắn gọn, bắt đầu bằng loại thay đổi:

```
feat: thêm kiểm tra thắng theo đường chéo
fix: sửa lỗi chặn hai đầu ở sát biên
test: thêm test hòa khi bàn đầy
docs: cập nhật README phần cài đặt
refactor: tách hàm lượng giá ra file riêng
```

Commit nhỏ, mỗi commit một ý. Lịch sử commit được dùng để chấm tỷ lệ đóng góp.

### Trước khi mở pull request

- [ ] Đã `git pull` `main` mới nhất và xử lý xung đột (nếu có).
- [ ] `pytest` chạy pass.
- [ ] Không commit file thừa (xem mục "Không đưa lên Git").
- [ ] Code mới có docstring ngắn cho hàm chính.

## Quy ước code

- Tuân theo **PEP 8**: tên hàm/biến `snake_case`, tên lớp `PascalCase`, hằng số `UPPER_CASE`.
- Tên biến, tên hàm bằng **tiếng Anh**; comment, docstring dùng tiếng Việt hoặc tiếng Anh đều được.
- Có **type hint** cho hàm công khai.
- **Không dùng số "ma thuật"**: kích thước bàn, độ sâu, số mô phỏng, seed… đặt trong `configs/` hoặc hằng số đầu file.
- Mọi thứ có ngẫu nhiên phải **nhận `seed`** để chạy lại ra đúng kết quả.

### Interface chung cho mọi tác tử

Mọi tác tử ở `agents/` đều kế thừa lớp này để arena và giao diện gọi được như nhau:

```python
class Agent:
    name: str = "base"

    def select_action(self, state, budget: float | None = None) -> int:
        """Trả về chỉ số ô 0–120 (hàng * 11 + cột).

        state:  trạng thái game hiện tại (không được sửa trực tiếp, dùng state.clone()).
        budget: thời gian suy nghĩ tối đa (giây) cho nước này; None = dùng mặc định.
        """
        raise NotImplementedError
```

- Tác tử **không được sửa** `state` truyền vào; cần thử nước đi thì dùng `clone()` hoặc `step()` + `undo()`.
- Trả về **nước hợp lệ**; nước bất hợp lệ sẽ làm engine ném ngoại lệ.

### Engine là phần dùng chung

- Mọi thay đổi trong `game/` phải có test kèm theo và báo trong nhóm.
- Từ ngày **8/11**, engine được **khóa** (tag `engine-v1.0`): chỉ sửa lỗi, không đổi luật hay API, để mọi kết quả giải đấu dùng cùng một phiên bản.

## Không đưa lên Git

Các file sau đã nằm trong `.gitignore`, **không commit**:

- Môi trường ảo: `.venv/`, `__pycache__/`
- Dữ liệu lớn, checkpoint, log thô: `data/`, `checkpoints/`, `logs/`, `*.pt`, `*.pth`, `*.npz`
- File riêng của máy: `.vscode/`, `.idea/`, `.DS_Store`

Checkpoint và dữ liệu lớn để trên **Google Drive chung của nhóm**; trong code chỉ ghi đường dẫn hoặc link tải.

## Lưu ý

- **Hạn nộp: 20/11/2026.** Theo dõi tiến độ ở tab Milestones.
- Gặp lỗi hoặc cần đổi interface thì **mở issue** hoặc nhắn nhóm trước, đừng tự sửa code của người khác.
- **Liêm chính học thuật:** code tham khảo từ sách (Mark Liu, *AlphaGo Simplified*, 2024), repo khác hay công cụ AI phải ghi nguồn ngay trong comment hoặc docstring.
