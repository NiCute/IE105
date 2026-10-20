# predict_cnn.py
# Dùng model Hybrid CNN đã train để dự đoán file csv mới
# Cách chạy:
#   python CNN_Model/predict_cnn.py --input file.csv --show 10
#
# Cần có 2 file cnn_best_model.pth và cnn_pipeline.joblib (chạy cnn_train.py sẽ tạo ra)
# Note: bản cũ phải đọc lại file csv gốc 11MB chỉ để lấy tên cột, chậm quá
# nên giờ lấy luôn tên cột đã lưu trong pipeline

import argparse
import sys
from pathlib import Path

# fix lỗi in tiếng Việt trên cmd windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))

try:
    from cnn_train import HybridMalwareModel, CLASS_NAMES
except ImportError:
    from CNN_Model.cnn_train import HybridMalwareModel, CLASS_NAMES

MODEL_PATH = BASE_DIR / "cnn_best_model.pth"
PIPELINE_PATH = BASE_DIR / "cnn_pipeline.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# nhãn hiển thị cho từng lớp (index giống CLASS_NAMES)
CLASS_TAG = {
    0: "[! Cảnh báo]",   # Adware
    1: "[! Nguy hiểm]",  # Banking
    2: "[! Nguy hiểm]",  # SMS
    3: "[! Rủi ro]",     # Riskware
    4: "[OK An toàn]",   # Benign
}


def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Không thấy file pipeline: {PIPELINE_PATH}. Chạy cnn_train.py trước đã.")
    return joblib.load(PIPELINE_PATH)


def load_model(n_features):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không thấy file model: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    # lúc train có lưu "mode" vào checkpoint rồi, đọc ra luôn chứ không fix cứng "hybrid"
    # (lỡ ai copy model_cnn.pth đè lên thì load_state_dict sẽ báo lỗi thiếu key rất khó hiểu)
    mode = ckpt.get("mode", "hybrid")
    model = HybridMalwareModel(n_features=n_features, n_classes=5, mode=mode).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parse_labels(raw_labels):
    # cột nhãn trong file test có thể là số (1-5 hoặc 0-4) hoặc là chữ
    # nên phải xử lý hết mấy trường hợp, cái nào không hiểu thì để -1
    #
    # BUG bản cũ: xét từng giá trị riêng lẻ, thấy 1..5 là trừ 1 luôn.
    # => file dùng nhãn 0..4 thì số 1 (Banking) bị đọc thành 0 (Adware), 2 -> 1,... lệch hết,
    #    chỉ có số 0 là đúng. Accuracy in ra sẽ sai bét mà không báo lỗi gì.
    # Sửa: phải nhìn CẢ CỘT mới biết file đang dùng kiểu nào.
    so = pd.to_numeric(pd.Series(raw_labels), errors="coerce")
    if so.notna().all():
        so = so.astype(int).values
        if so.min() == 0:
            # có số 0 -> chắc chắn là kiểu 0..4, giữ nguyên
            print("[NHÃN] Nhận diện cột nhãn kiểu số 0..4")
            ket_qua = so
        else:
            # không có số 0 -> coi là kiểu 1..5 giống file gốc CICMalDroid
            # (trường hợp file 0..4 mà không có mẫu Adware nào thì không phân biệt được, đành chịu)
            print("[NHÃN] Nhận diện cột nhãn kiểu số 1..5 (sẽ trừ 1)")
            ket_qua = so - 1
        # số nằm ngoài khoảng thì coi như không hiểu
        ket_qua = np.where((ket_qua >= 0) & (ket_qua < len(CLASS_NAMES)), ket_qua, -1)
        return ket_qua.astype(int)

    # không phải số hết thì đọc kiểu chữ
    name_to_idx = {
        "adware": 0,
        "banking": 1, "banking malware": 1,
        "sms": 2, "sms malware": 2, "smsmalware": 2,
        "riskware": 3,
        "benign": 4, "goodware": 4,
    }
    result = []
    for val in raw_labels:
        s = str(val).strip().lower()
        if s in name_to_idx:
            result.append(name_to_idx[s])
        else:
            result.append(-1)
    return np.array(result, dtype=int)


def preprocess_csv(csv_path, pipeline, label_col=None):
    df = pd.read_csv(csv_path)
    print(f"\n[TIỀN XỬ LÝ] Đọc file: {csv_path}")
    print(f"             Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")

    # tách cột nhãn ra nếu có (để tính accuracy)
    y_true = None
    if label_col and label_col in df.columns:
        y_true = df[label_col].values
        df = df.drop(columns=[label_col])
    elif "Class" in df.columns:
        y_true = df["Class"].values
        df = df.drop(columns=["Class"])

    train_features = pipeline.feature_names

    co = [f for f in train_features if f in df.columns]
    thieu = [f for f in train_features if f not in df.columns]

    print("[ĐỐI CHIẾU CỘT]")
    print(f"  - Model cần : {len(train_features)} cột")
    print(f"  - Khớp      : {len(co)} cột")
    print(f"  - Thiếu     : {len(thieu)} cột (cho bằng 0)")

    # tạo dataframe đúng thứ tự cột lúc train, cột nào thiếu thì để 0
    df_new = pd.DataFrame(0.0, index=df.index, columns=train_features)
    for col in co:
        df_new[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X_raw = df_new.values.astype(np.float32)

    X_raw_log = np.log1p(np.maximum(X_raw, 0))  # cho nhánh MLP
    X_img = pipeline.transform(X_raw)           # cho nhánh CNN, shape (N, 3, 24, 24)

    return X_img, X_raw_log, y_true


@torch.no_grad()
def predict_batch(model, X_img, X_raw_log, batch_size=64):
    t_img = torch.tensor(X_img, dtype=torch.float32)
    t_raw = torch.tensor(X_raw_log, dtype=torch.float32)

    all_probs = []
    all_preds = []
    # chạy theo từng batch cho đỡ tốn ram
    for i in range(0, len(t_img), batch_size):
        b_img = t_img[i:i + batch_size].to(DEVICE)
        b_raw = t_raw[i:i + batch_size].to(DEVICE)

        logits = model(b_img, b_raw)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_preds)


def print_results(probs, preds, y_true=None, topk=1, max_show=20):
    N = len(preds)
    print("\n" + "=" * 65)
    print(f"             KẾT QUẢ DỰ ĐOÁN ({N:,} MẪU)")
    print("=" * 65)

    y_true_arr = None
    if y_true is not None:
        y_true_arr = parse_labels(y_true)

    for i in range(min(N, max_show)):
        pred_class = int(preds[i])
        pred_name = CLASS_NAMES[pred_class]
        conf = probs[i][pred_class] * 100
        tag = CLASS_TAG[pred_class]

        true_str = ""
        if y_true_arr is not None:
            lbl = y_true_arr[i]
            if 0 <= lbl < len(CLASS_NAMES):
                if lbl == pred_class:
                    mark = "[ĐÚNG]"
                else:
                    mark = "[SAI]"
                true_str = f"  |  Nhãn gốc: {CLASS_NAMES[lbl]} {mark}"

        print(f"  Mẫu {i+1:>4}: {tag:<13} {pred_name:<20} ({conf:5.1f}% tin cậy){true_str}")

        # in thêm mấy lớp xếp sau nếu có --topk
        if topk > 1:
            top_idx = np.argsort(probs[i])[::-1][:topk]
            for rank in range(1, len(top_idx)):
                idx = top_idx[rank]
                print(f"                #{rank+1} {CLASS_NAMES[idx]:<20} ({probs[i][idx]*100:5.1f}%)")

    if N > max_show:
        print(f"  ... còn {N - max_show:,} mẫu nữa (thêm --show N để xem nhiều hơn)")

    # có nhãn thì tính luôn accuracy, f1
    if y_true_arr is not None:
        valid = (y_true_arr >= 0) & (y_true_arr < len(CLASS_NAMES))
        if valid.any():
            acc = accuracy_score(y_true_arr[valid], preds[valid])
            f1 = f1_score(y_true_arr[valid], preds[valid], average="macro")
            print("\n" + "=" * 65)
            print(f"  ACCURACY : {acc*100:.4f}%")
            print(f"  MACRO F1 : {f1:.4f}")
            print("=" * 65)
            print(classification_report(
                y_true_arr[valid], preds[valid],
                labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            ))


def main():
    parser = argparse.ArgumentParser(description="Dự đoán mã độc bằng Hybrid CNN")
    parser.add_argument("--input", required=True, help="đường dẫn file csv")
    parser.add_argument("--label", type=str, default="Class", help="tên cột nhãn (nếu có)")
    parser.add_argument("--topk", type=int, default=1, help="hiện top K lớp")
    parser.add_argument("--show", type=int, default=20, help="số mẫu in ra màn hình")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[LỖI] Không tìm thấy file: {args.input}")
        sys.exit(1)

    pipeline = load_pipeline()
    model = load_model(n_features=len(pipeline.feature_names))

    X_img, X_raw_log, y_true = preprocess_csv(str(input_path), pipeline, args.label)
    probs, preds = predict_batch(model, X_img, X_raw_log)

    print_results(probs, preds, y_true, args.topk, args.show)


if __name__ == "__main__":
    main()
