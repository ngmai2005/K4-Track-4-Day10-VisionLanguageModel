# Báo cáo lỗi và lưu ý — Lab Day 10

## Trạng thái môi trường đã kiểm chứng

Ngày chạy, dữ liệu được tải thành công từ Kaggle và `scripts/prepare_data.py --kaggle` sinh dataset version `d10-21e4070301`. Phân chia theo session tạo ra 1,422 ảnh train, 276 ảnh validation và 311 ảnh test; số box của từng lớp khớp với đặc tả đề bài.

`python -m pytest -q` đã cho kết quả `68 passed, 1 skipped`. Test bị skip là kiểm tra dữ liệu thật trước khi tải dataset nên là trạng thái dự kiến trước bước chuẩn bị dữ liệu.

## Lưu ý kỹ thuật bắt buộc

- Không chia ngẫu nhiên các frame dashcam: frame cùng session có bối cảnh gần trùng nhau và sẽ gây leakage. Dùng split do `prepare_data.py` sinh ra, không tự shuffle lại theo ảnh.
- Grounding phải dùng toạ độ số nguyên norm1000, không phải pixel 640×360. Box được sắp xếp `(y1, x1)` để mục tiêu autoregressive ổn định.
- Cấu hình NF4 phải giữ `visual` và `lm_head` ngoài lượng tử hoá. Các projector `visual.merger` và `visual.deepstack_merger_list` được mở ở E2/E3 phải dùng float32.
- Khi SFT, labels cho token ảnh và toàn bộ user prompt phải là `-100`; chỉ token câu trả lời assistant được tính cross-entropy loss.
- `pixel_values` và `image_grid_thw` của nhiều ảnh phải `torch.cat` dọc theo dim 0, không padding.
- PEFT chỉ lưu LoRA adapter. E2/E3 phải lưu và nạp lại riêng trọng số projector đã mở; nếu không kết quả inference sẽ quay về projector gốc.
- Không commit `data/`, `work/`, secrets, hoặc model weights. Không sửa `scripts/` và `tests/`.

## Sự cố đã gặp

Computer Use bị timeout sau khi đã tải và chia dữ liệu thành công. Đây là lỗi kết nối công cụ điều khiển, không phải lỗi Colab, T4 hay Kaggle. Không có số F1/VRAM/thời gian huấn luyện nào được ghi vào báo cáo khi chưa có evidence thực.
