"""
predict_hgb.py
==============
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Dự đoán phân loại mã độc Android cho TOÀN BỘ tệp dữ liệu CSV và tự động xuất file kết quả.
"""

import argparse
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from data_pipeline import FeaturePipeline

MODEL_PATH = BASE_DIR / "best_hgb_model.joblib"
PIPELINE_PATH = BASE_DIR / "pipeline.joblib"
TEST_DATA_PATH = BASE_DIR / "test_data.npz"

CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]


def main():
    parser = argparse.ArgumentParser(description="Dự đoán mã độc Android cho toàn bộ file CSV")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn tới file CSV cần dự đoán toàn bộ")
    parser.add_argument("--label", type=str, default=None, help="Tên cột nhãn thực tế nếu có (để tính Accuracy)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu file CSV kết quả dự đoán")
    parser.add_argument("--demo", action="store_true", help="Chạy thử nghiệm ngẫu nhiên 10 mẫu từ tập Test")
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        print("[LỖI] Chưa tìm thấy checkpoint mô hình hoặc pipeline đã lưu.")
        print("Vui lòng chạy 'python train_hgb.py' trước!")
        return

    print("=" * 65)
    print("      DỰ ĐOÁN TOÀN BỘ DỮ LIỆU BẰNG MÔ HÌNH HISTGRADIENTBOOSTING")
    print("=" * 65)
    print("[1/4] Đang nạp mô hình và pipeline...")
    model = joblib.load(MODEL_PATH)
    pipeline = FeaturePipeline.load(PIPELINE_PATH)

    # Chế độ DEMO: Lấy ngẫu nhiên mẫu từ tập test_data.npz
    if args.demo:
        if not TEST_DATA_PATH.exists():
            print("[LỖI] Không tìm thấy tệp test_data.npz để chạy Demo.")
            return

        print("\n--- CHẾ ĐỘ DEMO: DỰ ĐOÁN 10 MẪU NGẪU NHIÊN TỪ TẬP TEST ---")
        test_data = np.load(TEST_DATA_PATH, allow_pickle=True)
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]

        np.random.seed(42)
        indices = np.random.choice(len(y_test), size=10, replace=False)

        samples = X_test[indices]
        true_labels = y_test[indices]

        preds = model.predict(samples)
        probas = model.predict_proba(samples)

        print(f"{'STT':<4} | {'Thực tế':<12} | {'Dự đoán':<12} | {'Độ tin cậy':<12} | {'Trạng thái'}")
        print("-" * 60)
        for i in range(10):
            true_name = CLASS_NAMES[true_labels[i]]
            pred_name = CLASS_NAMES[preds[i]]
            conf = probas[i, preds[i]] * 100
            status = "✅ ĐÚNG" if true_labels[i] == preds[i] else "❌ SAI"
            print(f"{i+1:<4} | {true_name:<12} | {pred_name:<12} | {conf:6.2f}%      | {status}")
        return

    # Dự đoán file CSV người dùng chỉ định (mặc định lấy 471 nếu không truyền)
    input_path_str = args.input if args.input else "471_extracted features.csv"
    input_file = Path(input_path_str)
    if not input_file.exists():
        # Thử tìm trong thư mục Main
        input_file = BASE_DIR / input_path_str
        if not input_file.exists():
            print(f"[LỖI] Không tìm thấy file dữ liệu tại: {input_path_str}")
            return

    print(f"\n[2/4] Đang đọc toàn bộ dữ liệu từ: {input_file.name}")
    df_raw = pd.read_csv(input_file)
    n_samples = len(df_raw)
    print(f"  - Tổng số mẫu cần dự đoán: {n_samples:,} dòng x {df_raw.shape[1]} cột")

    # Tách cột nhãn nếu có
    y_true = None
    label_col = args.label
    if label_col is None and "Class" in df_raw.columns:
        label_col = "Class"

    if label_col and label_col in df_raw.columns:
        raw_labels = pd.to_numeric(df_raw[label_col], errors="coerce").fillna(0).values
        # Chuẩn hóa nhãn từ 1..5 về 0..4
        if raw_labels.min() >= 1:
            y_true = (raw_labels - 1).astype(int)
        else:
            y_true = raw_labels.astype(int)
        df_features = df_raw.drop(columns=[label_col])
    else:
        df_features = df_raw

    # 3. Tiền xử lý và dự đoán TOÀN BỘ
    print(f"\n[3/4] Đang chuyển đổi đặc trưng và dự đoán TOÀN BỘ {n_samples:,} mẫu...")
    X_trans = pipeline.transform(df_features)
    preds = model.predict(X_trans.values)
    probas = model.predict_proba(X_trans.values)
    print(f"  [V] Đã hoàn thành dự đoán 100% ({len(preds):,}/{n_samples:,} mẫu)!")

    # 4. Thống kê kết quả phân loại trên toàn bộ dữ liệu
    print("\n[4/4] BẢNG TỔNG HỢP KẾT QUẢ DỰ ĐOÁN TOÀN BỘ TẬP DỮ LIỆU:")
    print("-" * 65)
    unique_preds, counts_preds = np.unique(preds, return_counts=True)
    for u, c in zip(unique_preds, counts_preds):
        cname = CLASS_NAMES[u]
        pct = (c / n_samples) * 100
        print(f"  - [{cname:<12}]: {c:>6,} mẫu ({pct:5.2f}%)")
    print("-" * 65)

    # Lưu toàn bộ kết quả ra file CSV
    out_file = Path(args.output) if args.output else BASE_DIR / f"predictions_{input_file.stem}.csv"
    res_df = pd.DataFrame({
        "Index": np.arange(n_samples),
        "Predicted_Class": [CLASS_NAMES[p] for p in preds],
        "Confidence": np.max(probas, axis=1),
    })
    if y_true is not None:
        res_df["True_Class"] = [CLASS_NAMES[t] if 0 <= t < len(CLASS_NAMES) else "Unknown" for t in y_true]
        res_df["Is_Correct"] = (preds == y_true)

    for idx, cname in enumerate(CLASS_NAMES):
        res_df[f"Prob_{cname}"] = probas[:, idx]

    res_df.to_csv(out_file, index=False)
    print(f"\n💾 [ĐÃ XUẤT FILE] Toàn bộ {n_samples:,} dòng dự đoán đã được lưu tại:")
    print(f"   -> {out_file.resolve()}")

    # Nếu có nhãn thực tế, in báo cáo đánh giá chi tiết
    if y_true is not None:
        valid_mask = (y_true >= 0) & (y_true < len(CLASS_NAMES))
        if valid_mask.sum() > 0:
            acc = accuracy_score(y_true[valid_mask], preds[valid_mask])
            print("\n" + "=" * 65)
            print(f"🎯 ĐỘ CHÍNH XÁC TOÀN CỤC TRÊN TOÀN BỘ FILE: {acc * 100:.2f}%")
            print("=" * 65)
            print(classification_report(
                y_true[valid_mask],
                preds[valid_mask],
                target_names=CLASS_NAMES,
                digits=4,
                labels=list(range(len(CLASS_NAMES)))
            ))


if __name__ == "__main__":
    main()
