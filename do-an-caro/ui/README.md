# Giao diện demo Pygame

Chơi Caro 11×11 với các tác tử AI của đồ án, xem lại ván và xem thông số AI cân nhắc.

## Chạy

```bash
cd do-an-caro
pip install -r requirements.txt
python -m ui.app
```

## Chức năng

- **Người – Máy:** chọn cầm X (đi trước) hoặc O, chọn cấp AI đối thủ, click vào ô để đi.
  Nút **Hoàn tác** lùi về lượt của bạn (bỏ cả nước của AI).
- **Máy – Máy:** chọn cấp AI cho X và O, ván tự chạy; có **Tạm dừng / Tiếp tục** và **Đi 1 nước**.
- **Cấp AI 1–8:** cấp chưa có tác tử bị làm mờ ("chưa có"). Cấp 5–8 tự bật khi tác tử được
  đăng ký trong `AGENT_TYPES` của `arena/agent_factory.py` bằng đúng khóa ở `ui/levels.py`
  (`policy_net`, `rl_selfplay`, `puct`, `alphazero`); nếu nhóm đặt tên khác thì sửa bảng `LEVELS`.
- **Thời gian suy nghĩ:** 0,1 / 0,5 / 1,0 giây mỗi nước (tham số `budget` của tác tử).
- **AI cân nhắc:** ô vàng là nước vừa đi; khung bên phải hiện `last_stats` của nước AI gần nhất
  (số nút, độ sâu, số mô phỏng, điểm, thời gian...).
- **Xem lại ván:** nút `|<  <  >  >|`, **Phát** để tự phát; phím ← → Home End và Space.
  Khi xem lại, khung bên phải hiện thông số của đúng nước đang xem.

## Cấu trúc

| File | Nội dung |
| --- | --- |
| `app.py` | Cửa sổ pygame: menu, vẽ bàn cờ, nhận chuột/bàn phím |
| `session.py` | Ván cờ, hoàn tác, xem lại, tìm dãy thắng (không phụ thuộc pygame) |
| `ai_worker.py` | Chạy AI ở luồng nền để cửa sổ không bị đứng |
| `levels.py` | Danh sách cấp 1–8, thời gian suy nghĩ, tạo tác tử |
| `stats_format.py` | Định dạng `last_stats` thành tiếng Việt |

Kiểm thử: `pytest tests/test_ui_logic.py tests/test_ui_app.py` (giao diện chạy ở chế độ SDL không cửa sổ).
