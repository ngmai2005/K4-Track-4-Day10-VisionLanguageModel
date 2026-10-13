# Checklist — Lab Day 10

## Hoàn thành

- [x] Chọn runtime Google Colab T4.
- [x] Cài dependencies và kiểm tra đề bài: `68 passed, 1 skipped`.
- [x] Tải dữ liệu và chia theo session cố định.
- [x] Xác nhận `dataset_version`: `d10-21e4070301`.
- [x] Xác nhận split: train 1,422; val 276; test 311 ảnh.
- [x] Xác nhận phân bố box: train (1074/1750/691), val (66/470/103), test (121/299/163) cho pothole/crack/manhole.

## Cần chạy bằng Colab T4

- [ ] Hiện thực nhãn YOLO → grounding JSON chuẩn hoá 0–1000 và kiểm tra thứ tự `(y1, x1)`.
- [ ] Chạy zero-shot E0 và lưu `report/evidence/e0.json`.
- [ ] Hiện thực chat template, autoregressive loss mask và collator multimodal.
- [ ] Huấn luyện và đánh giá E1, E2, E3.
- [ ] Chạy ablation E4 với đúng một biến thay đổi.
- [ ] Lưu sáu hình dự đoán/ground truth và điền số liệu thật vào `REPORT.md`.
- [ ] Chạy `python scripts/check_submission.py` đến `0 FAIL`.
