# rf_train.py
# Train Random Forest làm baseline để so với CNN
# Dùng chung cách chia data với CNN (Common_Data/data_loader.py) cho công bằng

import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.append(str(Path(__file__).parent.parent))
from Common_Data.data_loader import get_train_val_test_data

# fix lỗi in tiếng Việt trên windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]
BASE_DIR = Path(__file__).parent.resolve()


def encode_label(y):
    # 1..5 -> 0..4, giống bên cnn_train.py
    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    return np.array([label_map[int(v)] for v in y])


def tao_rf(seed=42):
    # tách ra hàm riêng để run_ablation.py gọi lại với seed khác, tham số y chang
    # class_weight='balanced' vì số mẫu giữa các lớp chênh nhau
    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=seed,
        n_jobs=-1,  # chạy hết số core cho nhanh
        class_weight="balanced",
    )
    return rf


def main():
    print("=" * 60)
    print(" TRAIN RANDOM FOREST (BASELINE)")
    print("=" * 60)

    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw, _ = get_train_val_test_data()

    # RF thật ra không cần scale, nhưng để log1p cho giống input nhánh MLP bên CNN
    # (log là hàm đồng biến nên với cây quyết định cũng không ảnh hưởng mấy)
    X_train = np.log1p(np.maximum(X_train_raw, 0))
    X_val = np.log1p(np.maximum(X_val_raw, 0))
    X_test = np.log1p(np.maximum(X_test_raw, 0))

    y_train = encode_label(y_train_raw)
    y_val = encode_label(y_val_raw)
    y_test = encode_label(y_test_raw)

    rf = tao_rf(seed=42)

    print("\n[HUẤN LUYỆN] Đang fit Random Forest (150 cây)...")
    rf.fit(X_train, y_train)

    model_path = BASE_DIR / "rf_baseline_model.joblib"
    joblib.dump(rf, model_path)
    print(f"Đã lưu model: {model_path.name}")

    # xem thử trên val
    val_pred = rf.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, val_pred)*100:.4f}%")

    # test
    print("\n[ĐÁNH GIÁ TRÊN TẬP TEST]")
    test_pred = rf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred)*100:.4f}%")
    print(f"Test Macro F1: {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(classification_report(y_test, test_pred, target_names=CLASS_NAMES, digits=4))


if __name__ == "__main__":
    main()
