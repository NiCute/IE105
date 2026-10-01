# 🔬 BÁO CÁO PHÂN TÍCH SAI SỐ (ERROR ANALYSIS REPORT)
> **Đề tài:** Phân loại mã độc Android bằng mô hình CNN trên không gian ma trận ảnh 22×22
> **Tập dữ liệu:** CICMalDroid2020 Test Set (1,740 mẫu)

---

## 1. 📊 Tổng Quan Sai Số Toàn Cục

- **Tổng số mẫu kiểm thử:** 1,729 mẫu
- **Dự đoán chính xác:** 1,573 mẫu (90.98%)
- **Tổng số mẫu sai (Errors):** 156 mẫu (9.02%)
- **Độ tin cậy trung bình khi ĐÚNG:** 95.92%
- **Độ tin cậy trung bình khi SAI:** 66.17%
- **Số ca sai có độ tự tin cao (Confidence ≥ 70%):** 66 mẫu (42.3% tổng lỗi)

---

## 2. 🛡️ Đánh Giá Rủi Ro Bảo Mật (Security Impact)

| Loại rủi ro | Số mẫu | Tỷ lệ | Đánh giá an ninh |
|---|---|---|---|
| **False Negative (Lọt mã độc):** Malware bị đoán thành Benign | **20 mẫu** | **1.16%** | 🔴 **Nghiêm trọng:** Mã độc vượt qua hệ thống kiểm duyệt |
| **False Positive (Báo động giả):** Benign bị đoán thành Malware | **34 mẫu** | **1.97%** | 🟡 **Ảnh hưởng UX:** Ứng dụng lành tính bị chặn nhầm |
| **Lỗi nội bộ các dòng mã độc:** Malware loại A đoán thành B | **102 mẫu** | **5.90%** | 🟢 **Ít nguy hại:** Vẫn nhận diện được bản chất là mã độc |

---

## 3. 📈 Thống Kê Chi Tiết Sai Số Theo Từng Lớp

| Lớp (Class) | Tổng mẫu Test | Đúng | Sai | Recall (%) | Precision (%) |
|---|---|---|---|---|---|
| **Adware** | 188 | 180 | 8 | 95.74% | 64.75% |
| **Banking** | 307 | 278 | 29 | 90.55% | 94.24% |
| **SMS Malware** | 585 | 574 | 11 | 98.12% | 97.62% |
| **Riskware** | 380 | 306 | 74 | 80.53% | 97.76% |
| **Benign** | 269 | 235 | 34 | 87.36% | 92.16% |

---

## 3.1. 🧩 Phân Tích Độ Lệch Đặc Trưng Theo Nhóm Semantic
> So sánh cường độ kích hoạt trung bình giữa tập mẫu dự đoán **Đúng** và mẫu dự đoán **Sai**.

| Nhóm Semantic | Vùng Pixel | Số Feature | Đoán Đúng (Mean) | Đoán Sai (Mean) | Độ Lệch (Sai - Đúng) | Xu hướng sai lệch |
|---|---|---|---|---|---|---|
| **SystemCall** | `[0-138]` | 139 | 0.1881 | 0.1767 | **-0.0113** | 🔵 Kích hoạt yếu hơn bình thường |
| **Network** | `[139-161]` | 23 | 0.0415 | 0.0340 | **-0.0075** | 🔵 Kích hoạt yếu hơn bình thường |
| **Privacy** | `[162-182]` | 21 | 0.0188 | 0.0157 | **-0.0031** | 🔵 Kích hoạt yếu hơn bình thường |
| **Device** | `[183-231]` | 49 | 0.0496 | 0.0442 | **-0.0054** | 🔵 Kích hoạt yếu hơn bình thường |
| **Crypto** | `[232-238]` | 7 | 0.0002 | 0.0017 | **+0.0015** | 🔴 Kích hoạt bất thường cao hơn |
| **Process** | `[239-244]` | 6 | 0.1365 | 0.1289 | **-0.0076** | 🔵 Kích hoạt yếu hơn bình thường |
| **Android_Component** | `[245-288]` | 44 | 0.0076 | 0.0066 | **-0.0011** | 🔵 Kích hoạt yếu hơn bình thường |
| **App_Management** | `[289-313]` | 25 | 0.0176 | 0.0170 | **-0.0005** | 🔵 Kích hoạt yếu hơn bình thường |
| **FileSystem** | `[314-332]` | 19 | 0.1585 | 0.1451 | **-0.0134** | 🔵 Kích hoạt yếu hơn bình thường |
| **Other_API** | `[333-469]` | 137 | 0.0331 | 0.0329 | **-0.0002** | 🔵 Kích hoạt yếu hơn bình thường |

---

## 4. 🔀 Top Các Cặp Lớp Bị Nhầm Lẫn Nhiều Nhất

| Thứ hạng | Cặp nhầm lẫn (Thật → Đoán) | Số lượng | % trên tổng lớp thật | Nguyên nhân tiềm ẩn |
|---|---|---|---|---|
| #1 | **Riskware → Adware** | 52 mẫu | 13.68% | Hành vi tương đồng giữa các họ mã độc |
| #2 | **Benign → Adware** | 25 mẫu | 9.29% | Sử dụng API dùng chung giữa ứng dụng lành tính và mã độc |
| #3 | **Banking → Adware** | 18 mẫu | 5.86% | Hành vi tương đồng giữa các họ mã độc |
| #4 | **Riskware → Benign** | 10 mẫu | 2.63% | Sử dụng API dùng chung giữa ứng dụng lành tính và mã độc |
| #5 | **Riskware → Banking** | 8 mẫu | 2.11% | Hành vi tương đồng giữa các họ mã độc |
| #6 | **SMS Malware → Banking** | 6 mẫu | 1.03% | Hành vi tương đồng giữa các họ mã độc |

---

## 5. 🧠 Phân Tích Nguyên Nhân & Kiến Nghị Khắc Phục

### 5.1. Vì sao mô hình đưa ra dự đoán sai?
1. **Chồng lấn hành vi Semantic (Behavior Overlap):**
   - Các họ mã độc như *Riskware* và *SMS Malware* cùng sử dụng chung nhóm API Device (`SEND_SMS`, `READ_PHONE_STATE`, `GET_DEVICE_ID`).
   - Khi biểu diễn thành ảnh 22×22, các pixel tại khu vực *Device* và *Network* có cường độ kích hoạt rất gần nhau.
2. **Độ thưa của ma trận đặc trưng (Feature Sparsity):**
   - Nhiều mẫu chỉ kích hoạt một vài API đặc biệt thuộc nhóm `Other_API`, khiến phần lớn các vùng còn lại trên ảnh 22×22 gần bằng 0.
3. **Sự đánh đổi khi cân bằng lớp (Class Weight Trade-off):**
   - Việc áp dụng `class_weight` giúp cải thiện mạnh mẽ Recall của các lớp thiểu số (*Adware*, *Banking*), nhưng có thể làm tăng nhẹ một vài ca dự đoán nhầm giữa các lớp có trọng số tương tự.

### 5.2. Đề xuất cải tiến cho đồ án:
- **Focal Loss:** Thay thế hoặc kết hợp với CrossEntropy để giảm trọng số các mẫu dễ học (easy negatives) và tập trung vào các biên khó phân loại.
- **Attention Mechanism (SE-Net hoặc CBAM):** Bổ sung kênh chú ý để CNN tập trung vào các vùng Semantic quan trọng nhất (như vùng Crypto hoặc Privacy) thay vì quét đồng đều.
- **Sắp xếp thứ tự Feature tối ưu hơn:** Áp dụng thuật toán t-SNE hoặc MDS trên 470 features để các features có độ tương quan cao nằm cạnh nhau trên lưới 22×22.

---
*Báo cáo được khởi tạo tự động bởi mô đun `error_analysis.py` phục vụ bảo vệ đồ án IE105.*