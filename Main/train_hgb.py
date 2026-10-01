"""
train_hgb.py
============
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Kịch bản huấn luyện mô hình HistGradientBoosting kết hợp Feature Engineering.
"""

from pathlib import Path
import sys

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from data_pipeline import FeaturePipeline

DATA_PATH = BASE_DIR / "471_extracted features.csv"
MODEL_SAVE_PATH = BASE_DIR / "best_hgb_model.joblib"
PIPELINE_SAVE_PATH = BASE_DIR / "pipeline.joblib"
TEST_DATA_SAVE_PATH = BASE_DIR / "test_data.npz"

CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]


def main():
    print("=" * 65)
    print("        HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI MÃ ĐỘC ANDROID (IE105)")
    print("        Phương pháp: HistGradientBoosting + Feature Engineering")
    print("=" * 65)

    # 1. Đọc dữ liệu thô
    print("\n[1/5] Đang nạp dữ liệu từ:", DATA_PATH.name)
    df = pd.read_csv(DATA_PATH)
    n_initial = len(df)
    
    # Khử các dòng trùng lặp hoàn toàn
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  - Tổng số mẫu ban đầu: {n_initial}")
    print(f"  - Đã loại bỏ {n_initial - len(df)} dòng trùng lặp -> Còn lại: {len(df)} mẫu")

    feature_cols = [c for c in df.columns if c != "Class"]
    X_raw = df[feature_cols]
    y_raw = df["Class"].values

    # Ánh xạ nhãn: [1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4]
    y_encoded = y_raw - 1

    # 2. Phân chia chuẩn 70% Train - 15% Validation - 15% Test (Stratified)
    print("\n[2/5] Phân chia tập dữ liệu chuẩn đề cương (70% Train - 15% Val - 15% Test)...")
    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    print(f"  - Huấn luyện (Train) : {len(X_train_raw)} mẫu ({len(X_train_raw)/len(df)*100:.1f}%)")
    print(f"  - Kiểm định  (Val)   : {len(X_val_raw)} mẫu ({len(X_val_raw)/len(df)*100:.1f}%)")
    print(f"  - Kiểm thử   (Test)  : {len(X_test_raw)} mẫu ({len(X_test_raw)/len(df)*100:.1f}%)")

    # 3. Tiền xử lý và kỹ thuật đặc trưng (Fit DUY NHẤT trên tập Train)
    print("\n[3/5] Thực hiện quy trình Feature Engineering & Log-transform...")
    pipeline = FeaturePipeline()
    pipeline.fit(X_train_raw)

    X_train_trans = pipeline.transform(X_train_raw)
    X_val_trans = pipeline.transform(X_val_raw)
    X_test_trans = pipeline.transform(X_test_raw)

    print(f"  - Số đặc trưng sau khi xử lý: {X_train_trans.shape[1]} đặc trưng")
    print(f"  - Lưu bộ tiền xử lý tại: {PIPELINE_SAVE_PATH.name}")
    pipeline.save(PIPELINE_SAVE_PATH)

    # Lưu tập Test và Validation nén
    np.savez_compressed(
        TEST_DATA_SAVE_PATH,
        X_test=X_test_trans.values,
        y_test=y_test,
        X_val=X_val_trans.values,
        y_val=y_val,
        feature_names=np.array(pipeline.final_feature_cols),
    )
    print(f"  - Đã lưu tập Test & Validation tại: {TEST_DATA_SAVE_PATH.name}")

    # 4. Huấn luyện mô hình HistGradientBoosting
    print("\n[4/5] Đang huấn luyện HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.07,
        min_samples_leaf=25,
        max_leaf_nodes=45,
        l2_regularization=1.5,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train_trans.values, y_train)
    print(f"  - Mô hình hội tụ sau {model.n_iter_} vòng lặp!")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"  - Đã lưu checkpoint mô hình tại: {MODEL_SAVE_PATH.name}")

    # 5. Đánh giá trên tập Validation & Test
    print("\n[5/5] KẾT QUẢ ĐÁNH GIÁ (70% Train - 15% Val - 15% Test):")
    print("=" * 65)
    # Đánh giá tập Val
    y_pred_val = model.predict(X_val_trans.values)
    val_acc = accuracy_score(y_val, y_pred_val)
    print(f"📊 ĐỘ CHÍNH XÁC KIỂM ĐỊNH (VALIDATION ACCURACY) : {val_acc * 100:.2f}%")

    # Đánh giá tập Test
    y_pred_test = model.predict(X_test_trans.values)
    acc = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro")

    print(f"🎯 ĐỘ CHÍNH XÁC KIỂM THỬ   (TEST ACCURACY)       : {acc * 100:.2f}%")
    print(f"🎯 MACRO F1-SCORE                                : {macro_f1 * 100:.2f}%\n")
    print("--- BÁO CÁO CHI TIẾT TRÊN TẬP TEST ĐỘC LẬP ---")
    print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, digits=4))
    print("=" * 65)
    print("Hoàn tất huấn luyện! Để xuất đầy đủ đồ thị và báo cáo chi tiết, hãy chạy: python evaluate_hgb.py")


if __name__ == "__main__":
    main()
