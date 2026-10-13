# 📝 SỔ TAY BẢO VỆ ĐỒ ÁN IE105 — KHAI PHÁ DỮ LIỆU & ỨNG DỤNG

> **Chủ đề:** Phân loại mã độc Android 5 lớp (CICMalDroid2020)  
> **Hướng tiếp cận:** Mạng tích chập ảnh 22×22 (MalwareCNN) với vùng 139 System Call + Log-transform  
> *Ghi chú thực chiến, ngắn gọn, nói đúng bản chất, dễ nhớ khi lên bảng bảo vệ.*

---

## 💡 1. Câu chuyện đề tài tóm gọn trong 1 phút trình bày

1. **Vấn đề ban đầu:** 
   Nhóm chuyển 470 đặc trưng hành vi ứng dụng Android thành ảnh xám 22×22 để huấn luyện mạng CNN. Kết quả ban đầu chỉ đạt quanh **86%**.
2. **Nguyên nhân cốt lõi phát hiện được:**
   - Dữ liệu bị thưa nặng (**78% là số 0**) nhưng lại có ngoại lai cực lớn (**có hàm API bị gọi tới 3.6 triệu lần**).
   - Khi áp dụng chuẩn hóa Min-Max thông thường, những số triệu lần gọi này đã bóp nghẹt hơn 62% các tín hiệu kích hoạt nhỏ về sát 0 (biến thành các pixel đen xì, làm mất dấu vân tay hành vi).
   - Thứ tự các đặc trưng ban đầu bị xáo trộn ngẫu nhiên, khiến bộ lọc 3×3 của CNN quét trúng các API không liên quan.
3. **Giải pháp đột phá của nhóm:**
   - **Tách riêng 139 System Call** (`syscall_unique.csv`) đặt trọn vẹn vào **7 hàng đầu tiên** của ảnh ma trận (pixel 0 đến 138) để CNN học được "chữ ký kernel" liên tục.
   - **Áp dụng phép biến đổi Log-transform:** $\log(1 + x)$ trước khi chuẩn hóa Min-Max, giúp cân bằng độ tương phản cho bức ảnh.
   - **Kết quả:** Mạng CNN nhảy vọt từ **86.21% lên 90.98%** (Macro F1 đạt **89.10%**, ROC-AUC **0.9914**).

---

## 📊 2. Bảng so sánh "ăn tiền" nhất cho slide báo cáo

| Mô hình | Định dạng đầu vào | Accuracy | Macro F1 | Weighted F1 | Đánh giá thực tế |
|---|---|:---:|:---:|:---:|---|
| **CNN Gốc (Thô)** | Ảnh 22×22 xếp ngẫu nhiên | 86.21% | 84.26% | 86.35% | Bị ngoại lai triệu lần gọi làm đen kịt >62% điểm ảnh; kernel quét nhầm ranh giới. |
| **MalwareCNN (Đề xuất)** | Ảnh 22×22: **139 Syscall** [Hàng 0-6] + Log1p | **90.98%** | **89.10%** | **91.28%** | 🌟 **Bứt phá:** Gom 139 System Call giúp CNN học đúng không gian ngữ nghĩa; SMS đạt 97.9%, Banking 92.4%. |
| **MLP (Mạng Nơ-ron 1D)** | Bảng 470 vector + Chuẩn hóa | 92.54% | 91.12% | 92.65% | Mạng nơ-ron kết nối đầy đủ, không bị gò bó bởi lưới 2D nhân tạo nên học tốt hơn CNN gốc. |
| **Random Forest** | Bảng 470 vector + Log1p | 95.08% | 94.19% | 95.12% | Cây quyết định độc lập (Bagging), chạy rất nhanh (~0.5s) và phân loại tốt. |

---

## 🎯 3. Ba câu hỏi phản biện "tủ" thầy cô hay hỏi và cách trả lời

### ❓ Câu 1: *"Tại sao tách 139 System Call ra một vùng riêng lại giúp CNN tăng từ 86% lên 91%?"*
* **Trả lời trọng tâm:**
  > *"Thưa thầy/cô, CNN phát huy sức mạnh dựa trên giả định **tính tương quan không gian cục bộ (Spatial Locality)**: các pixel đứng cạnh nhau phải có liên kết với nhau. Ban đầu, các đặc trưng bị xếp lộn xộn nên kernel $3 \times 3$ quét qua các hành vi rời rạc.  
  > Khi nhóm nạp từ điển `syscall_unique.csv` (139 lời gọi kernel Linux) và xếp trọn vẹn vào 7 hàng đầu tiên (pixel 0 đến 138), bộ lọc tích chập đã thực sự học được 'dấu vân tay kernel' tập trung của mã độc. Kết hợp với phép biến đổi $\log(1+x)$ giải cứu các pixel không bị bóp về đen, CNN đã học được đúng các cụm hành vi và tăng vọt độ chính xác."*

### ❓ Câu 2: *"Tại sao CNN chỉ đạt 91% trong khi Random Forest trên bảng lại đạt 95%? Có phải do CNN dở hơn không?"*
* **Trả lời trọng tâm:**
  > *"Dạ thưa thầy/cô, không phải CNN dở, mà bản chất tập dữ liệu CICMalDroid2020 là **dữ liệu bảng dạng đếm tần suất (Tabular Count Data)** chứ không phải ảnh chụp quang học tự nhiên:  
  > 1. Các thuật toán cây (Random Forest) tách dữ liệu dựa trên các nhát cắt trực giao (ví dụ: `sendTextMessage >= 1`, `chmod > 0`), khớp hoàn hảo với các chỉ dấu kích hoạt mã độc.  
  > 2. Mạng CNN khi ép dữ liệu bảng thành lưới 2D sẽ vô tình tạo ra cấu trúc không gian giả tạo ở các ranh giới hàng/cột.  
  > 3. Tuy nhiên, hướng tiếp cận CNN dạng ảnh vẫn có giá trị đặc thù: trực quan hóa được 'vân tay mã độc' cho chuyên gia bảo mật nhìn thấy bằng mắt và có thể tận dụng phần cứng chip Edge-NPU trên điện thoại di động vốn tối ưu riêng cho phép tính tích chập 2D."*

### ❓ Câu 3: *"Bộ dữ liệu được tiền xử lý và chia tách thế nào để đảm bảo không bị học vẹt (Overfitting) hay rò rỉ dữ liệu (Data Leakage)?"*
* **Trả lời trọng tâm:**
  > *"Dạ thưa thầy/cô:  
  > - Nhóm đã lọc sạch **72 mẫu trùng lặp 100%** trong dữ liệu thô (từ 11,598 còn 11,526 mẫu).  
  > - Dữ liệu được chia theo tỷ lệ chuẩn **70% Train (8,068 mẫu) / 15% Validation (1,729 mẫu) / 15% Test (1,729 mẫu)** có phân tầng (`stratify=y`) để bảo toàn phân phối 5 lớp.  
  > - Bộ chuẩn hóa `MinMaxScaler` **chỉ được `fit()` duy nhất trên tập Train**, sau đó lưu thành file checkpoint để transform cho tập Val và Test. Tập Test hoàn toàn độc lập và chỉ được chấm điểm đúng 1 lần duy nhất sau cùng."*

---

## 🛠️ 4. Bảng hướng dẫn chạy lệnh nhanh (Cheat Sheet)

Tất cả lệnh đều chạy tại thư mục `d:\IE105\Main`:

| Mục đích | Lệnh chạy trong Terminal | Kết quả đầu ra |
|---|---|---|
| **Chạy Demo dự đoán thử mã độc (CNN)** | `python predict.py --demo` | In kết quả dự đoán 10 mẫu ngẫu nhiên kèm độ tin cậy (%) |
| **Xem biểu đồ đánh giá CNN** | `python evaluation.py` | Tạo đồ thị Confusion Matrix, ROC, Per-Class tại `evaluation/` |
| **Xem phân tích sai số chuyên sâu của CNN** | `python error_analysis.py` | Báo cáo chi tiết các ca CNN đoán nhầm tại `evaluation/` |
| **Tạo lại ảnh 22×22 từ CSV thô** | `python feature_to_image.py` | Nạp 139 syscall, xuất ra `images_22x22.npz` |
| **Huấn luyện lại CNN** | `python cnn_model.py` | Train 50 epoch PyTorch, lưu `best_model.pth` (90.98%) |

---

## 📂 5. Bản đồ thư mục `Main/` (Gọn gàng, sạch sẽ)

```text
Main/
├── 471_extracted features.csv   # Dữ liệu gốc CICMalDroid2020 (11,526 mẫu sau lọc)
├── syscall_unique.csv           # Danh mục 139 lời gọi hệ thống kernel Linux
├── GHI_CHU_DO_AN.md             # Sổ tay ghi chú này
│
└── [NHÁNH 1: MẠNG CNN 22×22]
    ├── feature_to_image.py      # Bộ chuyển đổi vector -> ảnh 22x22 (vùng 139 syscall + log1p)
    ├── images_22x22.npz         # Dữ liệu ảnh đã chia 70/15/15
    ├── scaler.joblib            # Bộ chuẩn hóa MinMaxScaler tập Train
    ├── cnn_model.py             # Mạng MalwareCNN (PyTorch)
    ├── best_model.pth           # Trọng số tốt nhất của CNN (90.98%)
    ├── evaluation.py            # Đánh giá CNN & xuất biểu đồ
    ├── error_analysis.py        # Phân tích ca sai của CNN
    ├── predict.py               # Dự đoán mẫu mới với CNN
    ├── visualize_samples.py     # Trực quan hóa bản đồ nhiệt ảnh 22x22
    └── evaluation/              # Thư mục chứa toàn bộ biểu đồ CNN
```
