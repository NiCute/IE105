# 📜 Lịch Sử Quá Trình Làm Đồ Án IE105 - Phân Loại Mã Độc Android

Tài liệu này tổng hợp lại toàn bộ các giai đoạn phát triển của đồ án tính đến thời điểm hiện tại, bao gồm các phương pháp đã dùng, các cải tiến cốt lõi và các bước đi tiếp theo.

---

## 🟢 Giai đoạn 1: Khởi tạo và Mô hình Baseline (Cơ sở)

*   **Task 1.1: Chuẩn bị dữ liệu và tiền xử lý ban đầu**
    *   **Phương pháp sử dụng:**
        *   **Min-Max Scaling:** Chuẩn hóa dữ liệu tuyến tính, đưa các giá trị đếm tần suất API trải dài về khoảng [0, 1].
        *   **Reshape 2D (Chuyển đổi 2D):** Ép 470 đặc trưng hành vi dạng bảng (tabular) thành một lưới ma trận kích thước 22x22 (tương đương 1 ảnh xám) với thứ tự sắp xếp ngẫu nhiên.
*   **Task 1.2: Huấn luyện mô hình CNN cơ sở (Baseline CNN)**
    *   **Phương pháp sử dụng:**
        *   **CNN (Mạng nơ-ron tích chập):** Sử dụng các bộ lọc (kernel) 3x3 để quét và học các đặc trưng cục bộ trên "bức ảnh" 22x22.
    *   **Kết quả:** Độ chính xác (Accuracy) đạt mức cơ bản khoảng **86.21%**.
    *   **Vấn đề phát hiện:** Dữ liệu bị thưa (78% là số 0) và có các giá trị ngoại lai cực lớn (có API bị gọi > 3.6 triệu lần). Chuẩn hóa Min-Max tuyến tính đã khiến các tín hiệu kích hoạt nhỏ bị bóp nghẹt thành 0 (ảnh bị đen kịt). Việc xếp ngẫu nhiên đặc trưng làm phá vỡ **Tính tương quan không gian cục bộ (Spatial Locality)** của CNN, khiến kernel 3x3 quét nhầm các hành vi không liên quan.

---

## 🟡 Giai đoạn 2: Cải tiến và Đột phá mô hình CNN (MalwareCNN)

*   **Task 2.1: Bố cục lại không gian ngữ nghĩa (Semantic-Spatial Matrix Layout)**
    *   **Phương pháp & Đề xuất:**
        *   **Tách riêng 139 Linux Kernel System Calls:** Trích xuất 139 lời gọi hệ thống cốt lõi và xếp trọn vẹn vào 7 hàng đầu tiên (pixel từ 0 đến 138) của ảnh 22x22. Đề xuất này giúp mạng CNN học được một "chữ ký kernel" tập trung và liên tục, giải quyết triệt để vấn đề ranh giới bị xáo trộn.
*   **Task 2.2: Xử lý ngoại lai và cân bằng tương phản ảnh**
    *   **Phương pháp & Đề xuất:**
        *   **Log-Transform Scaling (Biến đổi Logarit $\log(1+x)$):** Thay vì chỉ dùng Min-Max thông thường, áp dụng phép biến đổi logarit trước. Phép toán này giúp "nén" các giá trị ngoại lai hàng triệu lần xuống mức nhỏ, đồng thời "giải cứu" các giá trị nhỏ không bị biến mất về 0, làm tăng độ tương phản rõ rệt cho vân tay hành vi.
*   **Task 2.3: Huấn luyện lại và Đánh giá MalwareCNN**
    *   **Phương pháp sử dụng:** 
        *   **Stratified Split (Chia dữ liệu phân tầng):** Chia tập dữ liệu 70% Train / 15% Val / 15% Test, giữ nguyên tỷ lệ phân phối của 5 lớp mã độc.
    *   **Kết quả:** Độ chính xác tăng vọt lên **90.98%** (Macro F1: 89.10%, ROC-AUC: 0.9914). Đặc biệt bắt được rất tốt nhóm SMS Malware (97.87%) và Banking Trojan (92.36%).

---

## 🟠 Giai đoạn 3: Phát triển Mô hình Đối chuẩn (Benchmark) SOTA dạng bảng

*   **Task 3.1: Làm sạch dữ liệu và Kỹ thuật Đặc trưng (Feature Engineering)**
    *   **Phương pháp sử dụng:**
        *   **Data Deduplication (Lọc trùng lặp):** Loại bỏ 72 mẫu trùng lặp 100% trong dữ liệu thô để tránh hiện tượng rò rỉ dữ liệu (Data Leakage) sang tập Test.
        *   **Feature Engineering (Tạo đặc trưng mới):** Đề xuất tổng hợp thêm 4 nhóm chỉ số cấp cao dựa trên kiến thức chuyên gia, bao gồm:
            *   `feat_total_calls`: Tổng số lần gọi API trong suốt vòng đời.
            *   `feat_unique_calls`: Độ phong phú của API (số hàm khác nhau được gọi).
            *   Gom nhóm theo ngữ nghĩa (Privacy, Device, Network, Crypto).
            *   `feat_risk_ratio`: Tỷ lệ các hành vi nhạy cảm so với tổng thể (phản ánh mức độ nguy hiểm).
*   **Task 3.2: Huấn luyện mô hình Boosting**
    *   **Phương pháp & Đề xuất:**
        *   **HistGradientBoosting:** Một thuật toán học máy dạng cây (Tree-based) cực kỳ mạnh mẽ, tối ưu hóa rất tốt cho dữ liệu dạng bảng (Tabular Count Data). Thuật toán xây dựng các cây quyết định tuần tự, cây sau sẽ tập trung sửa lỗi cho cây trước. Phù hợp hoàn hảo với các nhát cắt dựa trên đếm tần suất hàm API.
    *   **Kết quả:** Đạt hiệu năng State-of-the-Art (SOTA - Tốt nhất) trong phạm vi đồ án với **95.66% Accuracy** (Macro F1: 94.75%).

---

## 🔵 Giai đoạn 4: Đánh giá, Trực quan hóa và Đóng gói (Hiện tại)

*   **Task 4.1: Phân tích sai số (Error Analysis) và Trực quan hóa**
    *   **Phương pháp sử dụng:**
        *   **Confusion Matrix & ROC Curve:** Sinh các đồ thị đánh giá để nhìn nhận khách quan tỷ lệ đoán đúng/sai cho từng họ mã độc.
        *   **Heatmap (Bản đồ nhiệt):** Vẽ trực quan các ma trận 22x22 bằng dải màu để hội đồng có thể thấy bằng mắt thường "vân tay" của từng loại malware.
*   **Task 4.2: Tự động hóa luồng xử lý (Pipeline) và Demo**
    *   **Phương pháp sử dụng:**
        *   Lưu toàn bộ các hàm chuẩn hóa (`scaler.joblib`) và trọng số (`best_model.pth`, `best_hgb_model.joblib`) được fit duy nhất trên tập Train.
        *   Viết các script chạy lệnh nhanh (`predict.py --demo`) để sẵn sàng biểu diễn live khi báo cáo.

---

## 🚀 Đề Xuất Các Bước Nên Làm Tiếp Theo (Cho các giai đoạn sau)

Để đồ án đạt điểm tối đa và mang tính thực tiễn cao hơn, dưới đây là các hướng đi tiếp theo:

1.  **Tối ưu hóa Siêu tham số (Hyperparameter Tuning):**
    *   **Mô tả:** Sử dụng các công cụ như `Optuna` hoặc `GridSearchCV` để dò tự động ra bộ tham số "vàng" (learning rate, số lớp mạng, max_depth, l2_regularization...). Hiện tại các thông số có thể mới được đặt theo kinh nghiệm (heuristic), việc tuning bài bản có thể đẩy accuracy mô hình Boosting lên > 96.5%.
2.  **Áp dụng AI Giải thích được (Explainable AI - XAI):**
    *   **Mô tả:** Tích hợp thư viện `SHAP` hoặc `LIME` cho mô hình Boosting và `Grad-CAM` cho mô hình CNN. Việc này giúp trả lời triệt để câu hỏi phản biện: *"Tại sao mô hình lại đoán đây là mã độc ngân hàng? Tính năng API nào quyết định điều đó?"* 
3.  **Thử nghiệm chéo trên dữ liệu mới (Cross-dataset Validation):**
    *   **Mô tả:** Tìm một tập dữ liệu malware Android khác (như Drebin hoặc AndroZoo) và nạp vào mô hình để xem khả năng chống chịu. Đánh giá **khả năng tổng quát hóa (Generalization)** xem mô hình có bị học vẹt (overfitting) với bộ dữ liệu CICMalDroid2020 hay không.
4.  **Triển khai thành Ứng dụng thực tế (Deployment / Web API):**
    *   **Mô tả:** Đóng gói mô hình `HistGradientBoosting` (vì độ trễ cực thấp) vào một Web API backend sử dụng `FastAPI` hoặc `Flask`. Tạo một giao diện Web đơn giản (hoặc App giả lập) cho phép người dùng upload một file CSV quét từ điện thoại lên, hệ thống sẽ trả về dự đoán Malware tức thời (Real-time Prediction).
5.  **Thu gọn mô hình CNN cho thiết bị di động (Edge AI & Quantization):**
    *   **Mô tả:** Chuyển đổi trọng số PyTorch sang định dạng di động như `ONNX` hoặc `TensorFlow Lite`. Dùng kỹ thuật Lượng hóa (Quantization) ép kiểu float32 xuống int8, làm giảm dung lượng file mô hình xuống chỉ còn vài chục KB. Chứng minh rằng ý tưởng dùng CNN 22x22 hoàn toàn có thể nhúng trực tiếp vào RAM của điện thoại Android để quét offline mà không cần mạng Internet.
