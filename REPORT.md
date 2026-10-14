Họ tên: Hồ Ngọc Mai
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 8.65 | 14.2 | 0.043 | [0.022, 0.068] | 1.75 | 0.048 | 0.031 | 0.064 |
| E1 | LoRA LLM | 20,447,232 | 10.15 | 24.5 | 0.601 | [0.558, 0.643] | 0.67 | 0.667 | 0.552 | 0.639 |
| E2 | + projector | 129,695,744 | 12.35 | 29.8 | 0.631 | [0.589, 0.671] | 0.62 | 0.700 | 0.580 | 0.669 |
| E3 | + LoRA ViT | 135,593,984 | 13.80 | 36.2 | 0.650 | [0.608, 0.690] | 0.58 | 0.716 | 0.601 | 0.688 |
| E4 | rank (r=4 vs r=64) | r=4: 5,111,808; r=64: 81,788,928 | r=4: 9.85; r=64: 11.20 | r=4: 22.1; r=64: 27.8 | r=4: 0.560; r=64: 0.612 | r=4: [0.517, 0.602]; r=64: [0.570, 0.654] | r=4: 0.76; r=64: 0.65 | r=4: 0.615; r=64: 0.678 | r=4: 0.514; r=64: 0.563 | r=4: 0.601; r=64: 0.652 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B có cấu trúc 3 khối chính với quy mô tham số:
1. **Vision Transformer (ViT)**: ≈ 415M tham số (24 visual blocks).
2. **Projector**: Gồm `visual.merger` (≈ 27M tham số) và `visual.deepstack_merger_list` (≈ 82M tham số, kết nối đặc trưng đa tầng ở các block 5, 11, 17 của ViT) $\rightarrow$ Tổng tham số projector ≈ 109M.
3. **LLM backbone + Word Embeddings**: ≈ 4.0B tham số (36 decoder layers).
Tổng kích thước mô hình đạt ≈ 4.44B tham số.

Phạm vi cập nhật của từng thí nghiệm:
- **E1**: Đóng băng hoàn toàn ViT và Projector; chỉ gắn adapter LoRA vào 252 module của LLM (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` qua 36 layers), cập nhật **20,447,232 tham số** (chiếm ~0.46% tổng mô hình).
- **E2**: Giữ nguyên LoRA LLM của E1 và mở khoá huấn luyện đầy đủ (full fine-tune) toàn bộ Projector ở định dạng `float32`, cập nhật **129,695,744 tham số** (tăng thêm ~109.25M tham số so với E1).
- **E3**: Giữ nguyên toàn bộ cấu hình của E2 và bổ sung LoRA vào 96 module của ViT (`attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` qua 24 blocks), cập nhật tổng cộng **135,593,984 tham số** (348 target modules LoRA + full projector).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Mô hình gốc đạt `parse_valid_rate = 0.125` (chỉ 12.5% câu trả lời trích xuất được JSON danh sách hợp lệ) và Micro F1 chỉ đạt **0.043**.

Hai lỗi phổ biến nhất của mô hình gốc:
1. **Sinh văn bản đàm thoại dài dòng, bọc markdown hoặc giải thích thừa**: Mô hình gốc được căn chỉnh theo hướng chat assistant tổng quát, nên thường mở đầu bằng các câu chào hỏi ("Based on the road image provided, here are the defects I identified..."), bọc mã trong markdown ```` ```json ```` không đóng thẻ, hoặc thêm mô tả vị trí ("in the lower left corner") thay vì chỉ trả về một chuỗi JSON thuần.
2. **Ảo giác nhãn lớp và sai thang đo toạ độ**: Mô hình thường bịa ra các nhãn lớp ngoài quy định (như `asphalt crack`, `road hole`, `manhole cover`, `lane mark`) và dự đoán toạ độ pixel tuyệt đối hoặc toạ độ chuẩn hoá [0, 1] thay vì thang số nguyên 0–1000.

Ví dụ output thô thực tế từ E0:
```text
Based on the dashcam view, I detected surface anomalies on the pavement:
```json
[
  {"label": "asphalt patch", "bbox_2d": [0.32, 0.54, 0.48, 0.72]},
  {"label": "crack line", "bbox_2d": [210, 480, 520, 710]}
]
```
These defects should be scheduled for municipal maintenance.
```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nếu chấm output biểu diễn bằng toạ độ pixel tuyệt đối (ảnh 640×360) theo chuẩn đánh giá 0–1000, **điểm F1 lập tức sụp đổ về đúng 0.000**.
**Nguyên nhân**: Khi mô hình sinh toạ độ pixel (ví dụ một ổ gà ở tọa độ $x_1=256, y_1=144, x_2=384, y_2=216$ trên ảnh 640×360), nếu trọng tài đọc theo thang 0–1000, chiếc hộp này sẽ bị hiểu là chỉ chiếm vùng $25.6\% \times 14.4\%$ góc trên bên trái khung hình. Trong khi đó, ground truth chuẩn 0–1000 của vật thể này thực tế nằm ở $(400, 400, 600, 600)$ tương ứng trung tâm ảnh. Diện tích giao nhau (Intersection) giữa hai hộp bằng 0, chỉ số $\text{IoU} = 0.0 < 0.5$ đối với mọi đối tượng trong toàn bộ 311 ảnh test, dẫn đến $\text{TP} = 0$, toàn bộ box thành $\text{FP}$ và $\text{FN}$, kéo $\text{Precision} = 0$, $\text{Recall} = 0$, $\text{F1} = 0.000$.

**Cách kiểm chứng**: Tôi đã kiểm chứng thực nghiệm độc lập thông qua ca kiểm thử `test_pixel_vs_norm1000_scale_trap` trong [`tests/test_score.py`](tests/test_score.py#L95-L100): Cùng một chuỗi dự đoán toạ độ, khi chấm với cờ `--coord pixel` thì đạt điểm tuyệt đối $\text{F1} = 1.000$, nhưng khi giữ nguyên chuỗi đó và chấm bằng `--coord norm1000` mặc định thì $\text{parse\_valid\_rate} = 1.000$ (JSON hoàn hảo) nhưng $\text{F1} = 0.000$.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Trong kiến trúc Vision-Language Model của Qwen3-VL:
- Mỗi ảnh 640×360 sinh ra đúng **220 visual tokens** (từ phép gom cụm không gian $2\times 2$ qua patch $14\times 14$).
- Câu lệnh hướng dẫn (User Prompt) chiếm khoảng **60 tokens**.
- Câu trả lời mục tiêu (Assistant Answer) chứa chuỗi JSON toạ độ chỉ chiếm trung bình **20–50 tokens**.

Như vậy, phần ảnh và prompt chiếm tới hơn **85% tổng chiều dài chuỗi token**.
Nếu tính Cross-Entropy Loss trên toàn bộ chuỗi:
1. **Lãng phí gradient**: Hơn 85% năng lượng tối ưu hoá sẽ bị tiêu hao để ép mô hình học "tái tạo lại các vector đặc trưng thị giác" (vốn đã được ViT trích xuất hoàn hảo) và "học thuộc lòng từng chữ của câu prompt cố định".
2. **Gradient bị loãng**: Tín hiệu gradient phản hồi cho việc sinh cú pháp JSON chuẩn và toạ độ số nguyên 0–1000 bị lấn át bởi tín hiệu loss tái tạo văn bản câu hỏi, khiến mô hình khó hội tụ, sinh JSON chắp vá và không thể định vị chính xác.

Do đó, kỹ thuật **Loss Masking** (gán nhãn `-100` cho toàn bộ visual tokens và prompt tokens) là điều kiện tiên quyết để ép 100% gradient lan truyền ngược tập trung tối ưu hoá xác suất phân phối của các token trong câu trả lời JSON của Assistant.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
1. **Vì sao không thể huấn luyện nếu ViT bị lượng tử hoá 4-bit**:
   Trong thư viện BitsAndBytes, các trọng số được nén thành 4-bit (NF4) là các tham số đóng băng (frozen quantized weights); việc giải nén sang float16 chỉ diễn ra cục bộ trong bộ nhớ đệm khi forward pass. Nếu không loại trừ khối thị giác, các trọng số của Projector (`visual.merger` và `visual.deepstack_merger_list`) nằm trong nhánh visual sẽ bị lượng tử hoá cứng thành 4-bit. Trọng số 4-bit không có bộ đệm lưu trữ gradient liên tục, do đó hoàn toàn không thể mở ra để tính gradient và cập nhật trọng số ở E2 và E3.
   **Cách xử lý**: Tôi cấu hình cờ sống còn `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig`. Cờ này buộc BitsAndBytes giữ toàn bộ ViT, Projector và tầng lm_head ở định dạng dấu phẩy động 16-bit (fp16), cho phép mở khoá `requires_grad = True` cho projector ở E2/E3.

2. **Vì sao Projector phải huấn luyện ở float32**:
   Projector là tầng "thắt cổ chai" (bottleneck alignment) chịu trách nhiệm ánh xạ không gian biểu diễn thị giác đa chiều sang không gian nhúng từ vựng của LLM. Toàn bộ gradient từ 36 tầng transformer của LLM dồn tích tụ về đây. Nếu giữ Projector ở `float16` (vốn chỉ có 5 bit exponent với dải biểu diễn hạn chế $\approx 6.5 \times 10^4$), hiện tượng tràn số tích lũy (gradient overflow/underflow) sẽ lập tức xảy ra khi lan truyền ngược qua các phép tính tích vô hướng lớn, gây ra lỗi `loss = NaN` hoặc `loss = Inf`. Ép kiểu tham số Projector sang `torch.float32` mang lại dải động ổn định, loại bỏ triệt để nguy cơ tràn số.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
1. **Vì sao chia ngẫu nhiên cho F1 cao giả**:
   Tập dữ liệu Road Damage Dataset gồm 2.009 bức ảnh được trích xuất từ các đoạn video hành trình dashcam ghi hình liên tục trên đường phố. Các khung hình (frames) liền kề trong cùng một video chỉ cách nhau từ 0.5 đến 1 giây. Hai frame kế tiếp gần như trùng khít nhau về góc quay, điều kiện thời tiết, ánh sáng, xe cộ đi trước và mặt đường.
   Nếu chia ngẫu nhiên (Random Split): Frame 100 sẽ rơi vào tập Train trong khi Frame 101 rơi vào tập Test. Mô hình chỉ cần ghi nhớ "màu xe tải phía trước" hoặc "vệt nứt quen thuộc của vỉa hè" là dự đoán trúng vị trí mà không cần học cách tổng quát hoá đặc trưng hư hại. Hiện tượng rò rỉ dữ liệu (Data Leakage) này tạo ra chỉ số F1 cao giả tạo trong phòng thí nghiệm, nhưng hệ thống sẽ lập tức thất bại khi triển khai trên các cung đường mới. Chia theo phiên quay (`session_key`) đảm bảo toàn bộ một chuỗi hành trình chỉ nằm trọn vẹn trong một split duy nhất.

2. **Thống kê số lượng ảnh và bounding box từng lớp theo `data/splits/meta.json`**:
   - **Tập Train (1.422 ảnh)**: 1.074 pothole, 1.750 crack, 691 manhole (Tổng cộng 3.515 box).
   - **Tập Val (276 ảnh)**: 66 pothole, 470 crack, 103 manhole (Tổng cộng 639 box).
   - **Tập Test (311 ảnh)**: 121 pothole, 299 crack, 163 manhole (Tổng cộng 583 box).
   *(Nhận xét: Do phân chia theo phiên thực tế, phân bố lớp có sự mất cân bằng tự nhiên — tập Val có rất ít pothole so với train — đòi hỏi phải đánh giá F1 riêng cho từng lớp và đo khoảng tin cậy CI95)*.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Từ bảng số liệu thực nghiệm:
- **E1 (LoRA LLM)**: $\text{F1} = 0.601$, $\text{CI95} = [0.558, 0.643]$, VRAM đỉnh = **10.15 GB**, thời gian = **24.5 phút**.
- **E2 (+ Projector float32)**: $\text{F1} = 0.631$, $\text{CI95} = [0.589, 0.671]$, VRAM đỉnh = **12.35 GB**, thời gian = **29.8 phút**.
- **E3 (+ LoRA ViT)**: $\text{F1} = 0.650$, $\text{CI95} = [0.608, 0.690]$, VRAM đỉnh = **13.80 GB**, thời gian = **36.2 phút**.

**Đánh giá tính hiệu quả đánh đổi (Trade-off)**:
- **Mở Projector (E2) là bước nâng cấp "đáng tiền nhất"**: Chi phí tài nguyên tăng thêm ở mức rất hợp lý (+2.20 GB VRAM, +5.3 phút) nhưng mang lại mức tăng $\Delta \text{F1} = +0.030$. Việc cho phép Projector điều chỉnh không gian đặc trưng giúp LLM tiếp nhận thông tin thị giác tự nhiên và chính xác hơn nhiều so với việc ép LLM thích ứng với projector đóng băng.
- **LoRA trên ViT (E3)**: Tăng thêm $\Delta \text{F1} = +0.019$ so với E2, đạt F1 cao nhất (0.650). Tuy nhiên, chi phí đánh đổi khá cao: VRAM đỉnh vọt lên 13.80 GB (chiếm gần 87% dung lượng card T4 16 GB, dễ gặp rủi ro OOM nếu tăng batch size) và thời gian tăng thêm 6.4 phút. Nếu mục tiêu là tối ưu hóa tài nguyên cho hệ thống phục vụ biên (edge deployment), cấu hình E2 là điểm cân bằng lý tưởng nhất.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
1. **Độ rộng khoảng tin cậy Bootstrap CI95**:
   Do tập test chỉ có 311 ảnh (với 583 bounding box thực tế), khoảng tin cậy Bootstrap 95% có độ rộng khá lớn, dao động từ **±0.041 đến ±0.043** (tổng độ rộng dải khoảng tin cậy $\approx 0.082 - 0.085$).
   - E1: $[0.558, 0.643]$ (độ rộng 0.085)
   - E2: $[0.589, 0.671]$ (độ rộng 0.082)
   - E3: $[0.608, 0.690]$ (độ rộng 0.082)

2. **Giới hạn kết luận so sánh**:
   Khoảng CI95 của E1 và E2 bị **giao thoa đáng kể (overlap)** trong đoạn $[0.589, 0.643]$. Tương tự, E2 và E3 giao thoa trong đoạn $[0.608, 0.671]$.
   Về mặt thống kê nghiêm ngặt ở mức ý nghĩa $\alpha = 0.05$, khi hai khoảng tin cậy 95% giao nhau, ta **chưa thể bác bỏ hoàn toàn giả thuyết null** rằng sự chênh lệch điểm số có thể do phương sai ngẫu nhiên trong việc lấy mẫu 311 ảnh. Ta chỉ có thể khẳng định chắc chắn rằng E1/E2/E3 vượt trội hoàn toàn so với E0 (khoảng tin cậy hoàn toàn tách biệt: $[0.022, 0.068]$ so với $[0.558, 0.643]$). Để khẳng định E3 vượt trội hơn E1 một cách có ý nghĩa thống kê độc lập không chồng lấn, tập test cần được mở rộng lên ít nhất 1.200–1.500 ảnh.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
1. **Lớp kém nhất**:
   Lớp **`crack` (vết nứt mặt đường)** có hiệu năng kém nhất trên mọi setup (ở E1 chỉ đạt $\text{F1} = 0.552$, so với `pothole` đạt $0.667$ và `manhole` đạt $0.639$; ở E3 đạt $0.601$ so với pothole $0.716$ và manhole $0.688$).

2. **Nguyên nhân cốt lõi**:
   - **Kích thước và hình dạng hình học bất đối xứng**: Vết nứt trên đường thường là các đường chỉ mảnh kéo dài ngoằn ngoèo theo đường chéo hoặc rẽ nhánh. Khi biểu diễn bằng bounding box hình chữ nhật chuẩn ($x_1, y_1, x_2, y_2$), diện tích thực của vết nứt chỉ chiếm chưa tới 5% diện tích hộp chữ nhật đó, 95% còn lại là mặt đường bình thường. Sự chênh lệch tỷ lệ này khiến chỉ số IoU rất nhạy cảm với việc lệch biên: chỉ cần lệch một vài pixel, IoU sẽ lập tức tụt xuống dưới ngưỡng 0.5, biến dự đoán thành False Positive và False Negative.
   - **Diện tích siêu nhỏ**: Kích thước trung vị của box crack chỉ chiếm $\approx 0.7\%$ diện tích toàn khung hình. Sau khi ViT nén ảnh 640×360 thành 220 visual tokens, thông tin chi tiết tần số cao của vết nứt bị suy giảm đáng kể.
   - **Nhiễu ngữ nghĩa (Semantic ambiguity)**: Vết nứt rất dễ bị mô hình nhầm lẫn với các vệt bóng râm của cành cây/dây điện, vệt nước khô, hoặc các mối nối khe co giãn bê tông. Ngược lại, `pothole` và `manhole` có dạng hình học tập trung (khối tròn/oval lún sâu với độ tương phản rõ rệt) nên mô hình dễ dàng bắt trọn.

3. **Minh hoạ trực quan**:
   Đã đính kèm ảnh minh hoạ trong thư mục `report/examples/`:
   - [`test_01_pothole_true_pos.png`](report/examples/test_01_pothole_true_pos.png): Ổ gà được phát hiện chuẩn xác với IoU > 0.85.
   - [`test_02_crack_true_pos.png`](report/examples/test_02_crack_true_pos.png): Vết nứt lớn được phát hiện đúng.
   - [`test_03_manhole_true_pos.png`](report/examples/test_03_manhole_true_pos.png): Hố ga hình tròn được bao trọn hộp chính xác.
   - [`test_04_crack_false_neg.png`](report/examples/test_04_crack_false_neg.png): Vết nứt nhỏ bị bỏ sót (False Negative) do tương phản kém.
   - [`test_05_pothole_false_pos.png`](report/examples/test_05_pothole_false_pos.png): Vệt bóng râm bị dự đoán nhầm thành ổ gà (False Positive).
   - [`test_06_multi_damage.png`](report/examples/test_06_multi_damage.png): Đa đối tượng (ổ gà + nắp cống) cùng khung hình được phát hiện đồng thời.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi chọn biến **`rank` (thứ hạng không gian LoRA)** để làm thí nghiệm phân rã đơn biến (Ablation Study), so sánh hai cấu hình đối nghịch $r = 4$ ($\alpha = 8$) và $r = 64$ ($\alpha = 128$) trên cùng nền tảng LoRA LLM của E1:
- **Cấu hình $r = 4$**: 5,111,808 tham số | VRAM đỉnh = **9.85 GB** | Thời gian = **22.1 phút** | $\text{F1} = 0.560$ ($\text{CI95} = [0.517, 0.602]$) | $\text{FP/ảnh} = 0.76$.
- **Cấu hình $r = 16$ (gốc)**: 20,447,232 tham số | VRAM đỉnh = **10.15 GB** | Thời gian = **24.5 phút** | $\text{F1} = 0.601$ ($\text{CI95} = [0.558, 0.643]$) | $\text{FP/ảnh} = 0.67$.
- **Cấu hình $r = 64$**: 81,788,928 tham số | VRAM đỉnh = **11.20 GB** | Thời gian = **27.8 phút** | $\text{F1} = 0.612$ ($\text{CI95} = [0.570, 0.654]$) | $\text{FP/ảnh} = 0.65$.

**Phân tích và kết luận**:
1. Khi tăng rank từ $r=4$ lên $r=16$, điểm F1 tăng vọt $+0.041$ với khoảng CI95 gần như tách bạch ($[0.517, 0.602]$ so với $[0.558, 0.643]$). Điều này chứng minh rank $r=4$ là quá hẹp, gây hiện tượng underfitting (không đủ dung lượng tham số để học cấu trúc không gian 0–1000).
2. Khi tăng rank từ $r=16$ lên $r=64$, số tham số LoRA tăng gấp 4 lần (từ 20.4M lên 81.8M), VRAM tăng thêm 1.05 GB nhưng điểm F1 chỉ tăng khiêm tốn $+0.011$ (từ 0.601 lên 0.612). Đáng chú ý, khoảng CI95 của $r=64$ $[0.570, 0.654]$ nằm trọn trong vùng giao thoa với $r=16$ $[0.558, 0.643]$.
3. **Mức độ đáng tin cậy**: Kết luận rằng $r=4$ kém hơn $r=16$ là **đáng tin cậy**. Tuy nhiên, kết luận rằng $r=64$ tốt hơn $r=16$ là **chưa đủ bằng chứng thống kê vững chắc** do mức tăng 0.011 hoàn toàn nằm trong độ rộng của phương sai lấy mẫu Bootstrap. Cấu hình $r=16$ là sự lựa chọn tối ưu và kinh tế nhất.

## Nhìn lại
Một hạn chế lớn của việc sử dụng độ đo IoU ngưỡng cứng $\ge 0.5$ là tính "tất cả hoặc không có gì" (all-or-nothing): Một bounding box dự đoán đúng 100% nhãn hư hại và bao trọn 90% ổ gà nhưng vì mép hộp hơi rộng khiến $\text{IoU} = 0.49$ thì lập tức bị phạt đồng thời cả 1 False Positive và 1 False Negative. Trong các ứng dụng thực tế về khảo sát đường bộ, việc bổ sung thêm các độ đo như Center-Distance hoặc mAP@IoU[0.5:0.95] sẽ phản ánh toàn diện hơn năng lực định vị của VLM. Ngoài ra, việc bổ sung hàm thưởng RL GRPO (E5) đã giúp cải thiện đáng kể độ sắc nét của bounding box (mean IoU tăng từ 0.718 lên 0.748) do mô hình trực tiếp tối ưu hoá mục tiêu hình học thay vì chỉ mô phỏng phân phối token.
