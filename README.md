# 🛡️ Phân Loại và Phát Hiện Mã Độc Android Bằng Mạng Nơ-ron Tích Chập (MalwareCNN)
## Android Malware Detection & Classification using 2D-CNN and Tabular Boosting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Topic: Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning%20%26%20Cybersecurity-red.svg)](https://github.com/)

> **Chuyên ngành:** An toàn Thông tin & Học sâu (Deep Learning in Cybersecurity)  
> **Bộ dữ liệu chuẩn:** CICMalDroid2020 (11,526 mẫu hành vi động sau tiền xử lý)  
> **Trọng tâm đề tài:** 
> 1. **Mô hình chính (Deep Learning):** Biểu diễn vân tay mã độc thành ảnh 2D ($22\times22$) với **vùng ưu tiên 139 Linux Kernel System Calls** và mạng **MalwareCNN (PyTorch)** (Đạt **90.98% Accuracy, Macro F1 89.10%, ROC-AUC 0.9914**).
> 2. **Mô hình đối chuẩn (Tabular Benchmark):** Mô hình cây **HistGradientBoosting** kết hợp 4 nhóm đặc trưng cấp cao (Đạt **95.66% Accuracy, Macro F1 94.75%**).

---

## 💡 1. Ý Tưởng & Đột Phá Kỹ Thuật

Phân loại mã độc từ 470 đặc trưng hành vi động khi chuyển thành ảnh 2D thường gặp 2 thách thức lớn:
- **Dữ liệu thưa (78% là số 0) & Ngoại lai cực đoan (có API gọi > 3.6 triệu lần):** Chuẩn hóa tuyến tính thông thường sẽ bóp nghẹt các tín hiệu kích hoạt nhỏ về 0.
- **Vi phạm tương quan không gian (Spatial Locality):** Sắp xếp đặc trưng ngẫu nhiên khiến kernel tích chập $3\times3$ quét nhầm các hành vi rời rạc.

### 🌟 Hai giải pháp cốt lõi:
1. **Log-Transform Scaling:** Áp dụng $\log(1+x)$ trước Min-Max, cân bằng độ tương phản cho bức ảnh và phục hồi các vân tay kích hoạt.
2. **Semantic-Spatial Matrix Layout:** Gom cụm trọn vẹn **139 lời gọi kernel Linux (`syscall_unique.csv`)** đặt vào **7 hàng đầu tiên (pixel 0–138)** của ma trận $22\times22$, sau đó sắp xếp 8 nhóm hành vi Android Framework (Network, Privacy, Device, Crypto...). Giúp CNN tăng vọt từ **86.21% lên 90.98%**.

---

## 📊 2. Bảng So Sánh Hiệu Năng (Test Set: 1,729 mẫu độc lập)

| Mô hình | Định dạng đầu vào | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | ROC-AUC | Vai trò trong đề tài |
|---|---|:---:|:---:|:---:|:---:|---|
| **Baseline CNN (Thô)** | Ảnh 22×22 ngẫu nhiên, MinMax | 86.21% | 84.26% | 86.35% | 0.9748 | Mô hình cơ sở ban đầu |
| **MLP (1D Nơ-ron)** | Vector 470 + MinMax | 92.54% | 91.12% | 92.65% | 0.9852 | Mạng nơ-ron kết nối đầy đủ |
| **Random Forest** | Vector 470 + Log1p | 95.08% | 94.19% | 95.12% | 0.9930 | Đối chuẩn cây độc lập (Bagging) |
| **MalwareCNN (Đề xuất)** | **Ảnh 22×22: 139 Syscall + Log1p** | **90.98%** | **89.10%** | **91.28%** | **0.9914** | 🌟 **Mô hình CNN trọng tâm của đề tài** |
| **HistGradientBoosting** | **Bảng sạch + Kỹ thuật đặc trưng** | **95.66%** | **94.75%** | **95.68%** | **0.9961** | 🏆 **Mốc đối chuẩn SOTA dạng bảng** |

---

## 🎯 3. Chi Tiết Hiệu Năng Theo Từng Họ Mã Độc

| Họ mã độc (Class) | MalwareCNN (22×22) F1 | HistGradientBoosting F1 | Đánh giá an ninh thực tế |
|---|:---:|:---:|---|
| **SMS Malware** | **97.87%** | **98.80%** | Bắt trọn gần 100% mã độc đánh cắp tin nhắn & mã OTP |
| **Banking Trojan** | **92.36%** | **94.55%** | Nhận diện chuẩn xác mã độc ngân hàng & lừa đảo tài chính |
| **Benign (Sạch)** | **89.69%** | **95.40%** | Tỷ lệ báo động giả (False Positive) rất thấp |
| **Riskware** | **88.31%** | **94.64%** | Phân loại chuẩn xác phần mềm tiềm ẩn nguy hiểm |
| **Adware** | **77.25%** | **90.36%** | Recall cao, không bỏ sót mã độc quảng cáo |

---

## 📂 4. Cấu Trúc Thư Mục Dự Án

```text
d:\IE105\
├── .gitignore                      # Chặn file > 100MB (50k sample) & cache Python
├── README.md                       # Tài liệu đồ án
│
├── dataset/                        # Dữ liệu nguồn
│   └── CSV/
│       ├── 50000_extracted features.csv  # Dữ liệu lớn CICMalDroid2020 (ignored git)
│       ├── 471_extracted features.csv    # Dữ liệu chuẩn 470 đặc trưng + nhãn Class
│       ├── 139_extracted features.csv    # Tập đặc trưng rút gọn 139
│       └── syscall_unique.csv            # Danh mục 139 lời gọi hệ thống kernel Linux
│
└── Main/                           # Toàn bộ mã nguồn & Model Checkpoints
    ├── 471_extracted features.csv  # Dữ liệu thực nghiệm chính
    ├── syscall_unique.csv          # Danh mục 139 syscall kernel
    ├── GHI_CHU_DO_AN.md            # Sổ tay bảo vệ đồ án & câu hỏi phản biện
    │
    ├── [Nhánh 1: MalwareCNN 22×22 (Trọng tâm đề tài)]
    │   ├── feature_to_image.py     # Biến đổi vector -> ma trận 22x22 (139 syscall + log1p)
    │   ├── images_22x22.npz        # File nén dữ liệu ảnh train/val/test (70/15/15)
    │   ├── scaler.joblib           # MinMaxScaler fit trên tập train
    │   ├── cnn_model.py            # Mạng PyTorch MalwareCNN (Conv2D, BatchNorm, Dropout)
    │   ├── best_model.pth          # Checkpoint trọng số CNN tốt nhất (90.98%)
    │   ├── evaluation.py           # Đánh giá & kết xuất đồ thị CNN (Confusion Matrix, ROC)
    │   ├── error_analysis.py       # Phân tích ca sai, độ tự tin và attribution
    │   ├── predict.py              # Dự đoán mã độc từ file CSV & chế độ Demo
    │   ├── visualize_samples.py    # Vẽ bản đồ nhiệt trực quan hóa ảnh 22x22
    │   └── evaluation/             # 17 báo cáo và đồ thị kết quả của CNN
    │
    └── [Nhánh 2: HistGradientBoosting (Đối chuẩn SOTA)]
        ├── data_pipeline.py        # Pipeline làm sạch và tạo 4 nhóm đặc trưng cấp cao
        ├── train_hgb.py            # Huấn luyện HistGradientBoosting (95.66%)
        ├── best_hgb_model.joblib   # Trọng số mô hình Boosting
        ├── pipeline.joblib         # Bộ trích chọn đặc trưng đã fit
        ├── test_data.npz           # Dữ liệu test chuẩn 1,729 mẫu của HGB
        ├── evaluate_hgb.py         # Đánh giá Boosting & xuất biểu đồ
        ├── predict_hgb.py          # Dự đoán & demo cho Boosting
        └── evaluation_hgb/         # Báo cáo đánh giá và biểu đồ của Boosting
```

---

## 🚀 5. Hướng Dẫn Chạy Nhanh (Cheat Sheet)

Chuyển vào thư mục `Main/` để thực thi:
```bash
cd Main
```

### Chạy Demo dự đoán thử nghiệm
```bash
# Demo với mô hình CNN (10 mẫu ngẫu nhiên từ Test set):
python predict.py --demo

# Demo với mô hình Boosting SOTA (95.66%):
python predict_hgb.py --demo

# Dự đoán file CSV mới bất kỳ:
python predict.py --input sample.csv --topk 3
```

### Xem biểu đồ đánh giá & phân tích sai số
```bash
# Đánh giá CNN (xuất Confusion Matrix, ROC-AUC vào evaluation/):
python evaluation.py

# Phân tích sai số chuyên sâu của CNN:
python error_analysis.py

# Trực quan hóa ảnh 22x22 và bản đồ nhiệt:
python visualize_samples.py

# Đánh giá mô hình Boosting:
python evaluate_hgb.py
```

### Huấn luyện lại từ đầu (Nếu cần)
```bash
# Huấn luyện CNN:
python feature_to_image.py
python cnn_model.py

# Huấn luyện Boosting:
python train_hgb.py
```
