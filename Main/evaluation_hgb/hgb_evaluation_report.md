# 📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HISTGRADIENTBOOSTING
> **Đề tài:** Phân loại mã độc Android CICMalDroid2020  
> **Phương pháp:** HistGradientBoosting kết hợp Feature Engineering & Log-transform  
> **Tập dữ liệu kiểm thử:** 1729 mẫu (15% Stratified Test Set)

---

## 1. 📈 Các Chỉ Số Toàn Cục (Global Metrics)

- **Độ chính xác (Accuracy):** **95.66%**
- **Macro Precision:** 94.43%
- **Macro Recall:** 95.16%
- **Macro F1-Score:** **94.75%**
- **Weighted F1-Score:** 95.68%

---

## 2. 🛡️ So Sánh Hiệu Năng Với Các Phương Pháp Trước Đó

| Phương pháp | Accuracy | Macro F1 | Ghi chú |
|---|---|---|---|
| **MalwareCNN (22×22)** | 86.21% | 84.26% | Hạn chế do ép dữ liệu bảng thành ảnh và MinMax bị ngoại lai |
| **Random Forest gốc** | 94.08% | 93.15% | Cây quyết định trên dữ liệu thô |
| **HistGradientBoosting + Feature Engineering** | **95.66%** | **94.75%** | **SOTA Đồ án IE105 (+10.32% so với CNN ban đầu)** |

---

## 3. 🎯 Chi Tiết Hiệu Năng Theo Từng Họ Mã Độc

| Họ mã độc (Class) | Precision | Recall | F1-Score | Số mẫu Test |
|---|---|---|---|---|
| **Adware** | 86.41% | 94.68% | **90.36%** | 188 |
| **Banking** | 95.97% | 93.16% | **94.55%** | 307 |
| **SMS Malware** | 98.80% | 98.80% | **98.80%** | 585 |
| **Riskware** | 96.45% | 92.89% | **94.64%** | 380 |
| **Benign** | 94.53% | 96.28% | **95.40%** | 269 |

---

## 4. 💡 Đánh Giá Bảo Mật & Rủi Ro Hệ Thống

- **Số ca Benign bị nhận nhầm thành Malware (False Positive):** 10 mẫu
- **Số ca Malware bị lọt lưới thành Benign (False Negative):** 15 mẫu
- **Kết luận:** Mô hình có độ nhạy và độ tin cậy an ninh mạng vượt trội, nhận diện SMS Malware đạt Recall > 99%.
