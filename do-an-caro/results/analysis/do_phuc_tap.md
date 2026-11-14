# Đặc điểm và độ phức tạp của Caro 11×11 luật Việt Nam

| đại lượng | giá trị | cách tính |
|---|---|---|
| số ô / số hành động tối đa | 121 | bàn 11×11, mỗi ô trống là một nước hợp lệ |
| cận trên số trạng thái | ≈ 10^56.7 | Σ_k C(121,k)·C(k,⌈k/2⌉): số cách đặt k quân với số X bằng số O hoặc hơn 1, chưa loại thế đã kết thúc |
| cận thô 3^121 | ≈ 10^57.7 | mỗi ô trống/X/O |
| độ dài ván trung bình (không tính ván có Random) | 19.0 nước (trung vị 14, tối đa 121) | 6.600 ván chế độ B, gồm 4 nước khai cuộc |
| ước lượng cây trò chơi | ≈ 10^38.9 | tích số nước hợp lệ 121·120·…·103 theo độ dài ván trung bình 19 |
| tỷ lệ hòa | 66/10800 ván chế độ B (0.6%) | hòa chỉ khi đủ 121 ô |
| lợi thế đi trước | score X = 0.520 (chế độ B); X thắng 57.2% trong 20.000 ván sinh dữ liệu | |

Tính chất: hai người chơi, quan sát đầy đủ, tất định, tổng bằng không (thắng +1, thua −1, hòa 0 theo góc nhìn người chơi).
