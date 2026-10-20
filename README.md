# Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
**Phát hiện và Phân loại Mã độc Android bằng Kỹ thuật Deep Learning (Kiến trúc Hybrid)**

Chào mừng đến với dự án cuối kỳ môn Khai phá dữ liệu (IE105). Repo này chứa mã nguồn để phát hiện và phân loại mã độc Android dựa trên các hành vi (lời gọi hệ thống Linux, API calls, quyền hạn, intent, phần cứng...) được trích xuất từ môi trường sandbox động (bộ dữ liệu CICMalDroid2020).

---

## 📌 Giới thiệu dự án

Mã độc Android ngày càng tinh vi và sử dụng nhiều kỹ thuật lẩn tránh. Thay vì chỉ phân tích tĩnh file APK (dễ bị obfuscation qua mặt), chúng ta thu thập **470 đặc trưng hành vi động**:
- **139 Lời gọi hệ thống cốt lõi (Linux Kernel System Calls)**: `open`, `read`, `write`, `clone`, `futex`, `ioctl`...
- **331 Đặc trưng hành vi tầng Android**: Mạng (Socket, HTTP, DNS), Quyền hạn nhạy cảm (GPS, Danh bạ, Camera), Phần cứng (SMS, Phone, Wifi), Mã hóa (Crypto/Cipher), Tiến trình và Các thành phần Android (Intent, Service, Receiver).

Thử thách đặt ra là: **Làm sao để đưa dữ liệu dạng bảng (tabular data) vào mạng nơ-ron (CNN) vốn được thiết kế cho xử lý ảnh, mà vẫn giữ được độ chính xác cạnh tranh với các mô hình cây quyết định (Tree-based) như Random Forest?**

Nhóm giải quyết bài toán này bằng kiến trúc **Hybrid CNN + 1D MLP**: kết hợp ảnh 2D ngữ nghĩa (xử lý qua CNN) và vector đặc trưng toàn cục (xử lý qua MLP).

---

## 🚀 Các Cải Tiến Kỹ Thuật

Để mô hình CNN có thể học tốt trên dữ liệu bảng, nhóm áp dụng các kỹ thuật sau:

1. **Gom nhóm ngữ nghĩa (Semantic Grouping) & Căn hàng ma trận 24x24**:
   - Dùng Regex tokenization chia 470 đặc trưng vào 10 nhóm chức năng (SystemCall, Network, Privacy, Device, Crypto, Process, Android_Component, App_Management, FileSystem, Other_API).
   - Mỗi nhóm bắt đầu ở một hàng mới trên ma trận 24x24 (tổng 576 ô, dùng 23/24 hàng). Phần dư ở cuối hàng được đệm số 0 (Zero-padding) để một hàng không chứa feature của 2 nhóm khác nhau.
   - *Lưu ý:* việc này chỉ tránh trộn nhóm **theo chiều ngang**. Kernel 3x3 phủ 3 hàng liền nhau nên ở ranh giới giữa 2 nhóm vẫn quét lẫn **theo chiều dọc** (không đủ hàng trống để chèn hàng phân cách giữa 10 nhóm). Ngoài ra nhóm `Other_API` vẫn chiếm 152/470 feature do bộ từ khóa chưa phủ hết.

2. **Sắp xếp theo Mutual Information (Thông tin tương hỗ)**:
   - Thay vì dùng hệ số tương quan tuyến tính Pearson (vốn giả định nhãn là số liên tục, không hợp lý với nhãn phân loại 5 lớp), nhóm sử dụng `mutual_info_classif` để đo lường mức độ liên quan phi tuyến giữa từng đặc trưng với nhãn mã độc. Trong mỗi nhóm ngữ nghĩa, đặc trưng có điểm Mutual Information cao hơn sẽ được xếp trước.

3. **Biểu diễn ảnh đa kênh (3-Channel Image)**:
   - **Kênh 1 (Cường độ)**: Log-transform `log(1+x)` kết hợp Min-Max Scaling (nén các giá trị gọi hàng triệu lần, cứu các tín hiệu nhỏ).
   - **Kênh 2 (Nhị phân)**: Cờ Có/Không kích hoạt (0/1), mô phỏng cơ chế rẽ nhánh của cây quyết định.
   - **Kênh 3 (Chuẩn hóa nội bộ mẫu)**: Sample-wise Normalization (chia cho giá trị max của chính mẫu đó).

4. **Giữ thông tin vị trí không gian (Spatial Awareness)**:
   - Không sử dụng Global Average Pooling (GAP) vì GAP nén toàn bộ ma trận về 1x1 làm mất thông tin vị trí các hàng ngữ nghĩa.
   - Thay bằng `AdaptiveAvgPool2d((3, 3))` và bổ sung module `CoordConv` (ghép thêm 2 kênh tọa độ X, Y vào ảnh).

5. **Kiến trúc Hybrid song song (CNN 2D + 1D MLP)**:
   - Nhánh CNN khai thác các mối tương quan cục bộ trong ma trận 2D.
   - Nhánh MLP (2 lớp Linear + BatchNorm + Dropout) tiếp nhận trực tiếp vector 470 đặc trưng gốc để bù đắp các liên kết toàn cục mà cửa sổ tích chập 3x3 bỏ sót. Hai luồng được ghép (concatenate) trước khi đưa vào Classifier 5 lớp.

6. **Huấn luyện cân bằng & Chống Overfitting**:
   - Loại bỏ mẫu trùng lặp hoàn toàn **trước khi chia tập** (tránh cùng 1 mẫu vừa nằm ở Train vừa nằm ở Test → data leakage) và loại bỏ mẫu xung đột (cùng vector đặc trưng nhưng khác nhãn → nhiễu nhãn). Tổng cộng bỏ 80/11,598 mẫu.
   - Áp dụng Class Weight (căn bậc 2) kết hợp Label Smoothing 0.1 và chu kỳ học `OneCycleLR`. Lưu checkpoint theo **Macro-F1** trên tập Validation.

---

## 📂 Cấu trúc thư mục

```text
IE105
├── Common_Data/
│   ├── 471_extracted features.csv   # Dữ liệu gốc 470 đặc trưng + 1 cột nhãn Class
│   ├── syscall_unique.csv           # Danh sách 139 System Calls
│   └── data_loader.py               # Làm sạch, chia Train/Val/Test (70/15/15) chung
├── CNN_Model/
│   ├── preprocess.py                # Pipeline biến đổi Vector -> Tensor ảnh 24x24
│   ├── cnn_train.py                 # Huấn luyện mô hình Hybrid / CNN / MLP
│   ├── predict_cnn.py               # Script dự đoán độc lập file CSV mới
│   ├── test_grouping.py             # Script kiểm tra căn hàng ma trận ngữ nghĩa
│   ├── cnn_best_model.pth           # Trọng số mô hình tốt nhất
│   └── cnn_pipeline.joblib          # Pipeline ánh xạ đặc trưng đã fit trên Train
├── RandomForest_Baseline/
│   ├── rf_train.py                  # Huấn luyện mô hình đối chuẩn Random Forest
│   └── rf_baseline_model.joblib     # File mô hình Random Forest
├── run_ablation.py                  # Script chạy thử nghiệm Ablation Study (5 seeds)
├── ablation_results.csv             # Kết quả chi tiết các lần chạy ablation
├── cnn_log.txt                      # Log kết quả huấn luyện mô hình Hybrid CNN
├── rf_log.txt                       # Log kết quả huấn luyện mô hình Random Forest
└── README.md
```

---

## 🧠 Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Cần Python 3.9+ và các thư viện trong danh sách sau:
```bash
pip install pandas numpy scikit-learn torch joblib
```

### 2. Huấn luyện Random Forest (Baseline)
```bash
python RandomForest_Baseline/rf_train.py
```
*(Chạy mất khoảng 5-10 giây trên CPU, lưu model tại `RandomForest_Baseline/rf_baseline_model.joblib`).*

### 3. Huấn luyện mạng Hybrid CNN
```bash
python CNN_Model/cnn_train.py
```
*(Chạy khoảng 4-5 phút trên CPU, lưu checkpoint và pipeline tại thư mục `CNN_Model/`).*

### 4. Chạy kiểm nghiệm Ablation Study
Để kiểm chứng đóng góp của từng thành phần (CNN-only, MLP-only, Hybrid, Random Forest) qua 5 seed ngẫu nhiên:
```bash
python run_ablation.py
```
Kết quả từng lượt chạy sẽ được tự động lưu vào `ablation_results.csv`.

### 5. Dự đoán trên tập dữ liệu mới
Công cụ dự đoán độc lập, tự động căn chỉnh và xử lý cột thiếu:
```bash
python CNN_Model/predict_cnn.py --input "Common_Data/471_extracted features.csv" --show 10
```
*(Lưu ý: file 471 gốc chứa cả dữ liệu Train nên accuracy in ra ở lệnh trên sẽ cao hơn thực tế, chỉ dùng để demo. Con số đánh giá chính thức là trên tập Test bên dưới. Cột nhãn kiểu số 1..5, 0..4 hoặc tên lớp đều đọc được.)*

---

## 📊 Kết quả thực nghiệm

Đánh giá trên tập kiểm thử độc lập (**Test Set: 1,728 mẫu** hoàn toàn chưa từng xuất hiện trong quá trình huấn luyện và chọn siêu tham số):

### 1. Kết quả kiểm nghiệm Ablation Study (Đánh giá qua 5 Seeds ngẫu nhiên)

Bảng tổng hợp thực nghiệm đa seed (`[42, 43, 44, 45, 46]`) nhằm loại bỏ yếu tố may rủi, phản ánh chính xác hiệu năng mô hình (báo cáo Mean ± Std).
*Lưu ý: seed chỉ thay đổi khởi tạo trọng số / thứ tự shuffle (RF: bootstrap), còn cách chia Train/Val/Test cố định `random_state=42` nên Std chưa phản ánh dao động do chia dữ liệu.*

| Mô hình | Số Seeds | Test Accuracy (%) | Test Macro F1 | Thời gian train TB (CPU) | Điểm nổi bật |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest (Baseline)** | 5 | **95.90 ± 0.08** | **0.9515 ± 0.0011** | ~0.6s | Độ ổn định gần như tuyệt đối, cực kỳ mạnh trên bảng rời rạc |
| **Hybrid CNN + MLP** | 5 | **95.64 ± 0.32** | **0.9475 ± 0.0036** | ~280s | Kết hợp đa kênh và vector 1D, tiệm cận sát mốc của Random Forest |
| **MLP-only (1D Vector)** | 5 | **95.68 ± 0.37** | **0.9480 ± 0.0040** | ~26s | Xử lý nhanh, khai thác triệt để tương quan toàn cục của vector 470 |
| **CNN-only (2D Image)** | 5 | **95.36 ± 0.18** | **0.9449 ± 0.0021** | ~275s | Độ lệch chuẩn nhỏ (±0.18%), chứng minh ma trận 24x24 học rất ổn định |

*(Toàn bộ log chi tiết từng seed được lưu tại file `ablation_results.csv`).*

### 2. Chi tiết theo từng lớp mã độc (Mô hình Hybrid CNN - Seed 42, đây là seed thấp nhất của Hybrid)

```text
              precision    recall  f1-score   support
      Adware     0.8683    0.9468    0.9059       188
     Banking     0.9754    0.9085    0.9408       306
 SMS Malware     0.9798    0.9932    0.9864       585
    Riskware     0.9558    0.9105    0.9326       380
      Benign     0.9187    0.9665    0.9420       269
    accuracy                         0.9508      1728
   macro avg     0.9396    0.9451    0.9415      1728
```

---

## 💡 Nhận xét & Đánh giá khoa học

1. **Về bản chất dữ liệu**: Dữ liệu hành vi mã độc dạng bảng có cấu trúc rời rạc, tính thưa cao (nhiều giá trị 0) và các ranh giới quyết định trực giao. Đây là môi trường sở trường của các thuật toán họ Cây (Random Forest, Gradient Boosting).
2. **Hiệu quả của việc biến đổi ảnh**: Việc sắp xếp đặc trưng theo nhóm ngữ nghĩa và căn hàng 24x24 kết hợp tính điểm bằng Mutual Information đã giúp CNN đạt **95.36%** (trung bình 5 seed) độc lập, vượt xa mức 86% ban đầu khi xếp pixel ngẫu nhiên (lưu ý: bản 86% dùng bộ dữ liệu/tiền xử lý cũ nên không hoàn toàn cùng điều kiện).
3. **Ý nghĩa của kiến trúc lai (Hybrid)**: Kết quả ablation cho thấy **MLP-only (95.68 ± 0.37) và Hybrid (95.64 ± 0.32) gần như bằng nhau**, chênh lệch nhỏ hơn nhiều so với độ lệch chuẩn → chưa có bằng chứng nhánh CNN đóng góp thêm thông tin so với vector 1D. Cách giải thích hợp lý: hai ô kề nhau trên ảnh chỉ là 2 feature có điểm MI gần nhau, không nhất thiết có quan hệ với nhau như pixel ảnh thật, nên tính "cục bộ không gian" mà CNN khai thác khá yếu. Random Forest vẫn tốt nhất cả về độ chính xác, độ ổn định và thời gian train.
