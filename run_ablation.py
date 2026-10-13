# run_ablation.py
# Chạy ablation: so sánh 3 kiểu model (hybrid, chỉ CNN, chỉ MLP) + Random Forest, mỗi cái 5 seed
# Mục đích: trả lời câu "nhánh CNN có thực sự giúp gì không hay chỉ mình MLP là đủ?"
# và xem chênh lệch giữa các model có lớn hơn mức dao động do random không.
#
# Cách chạy:
#   python run_ablation.py
# File tự động lưu kết quả vào ablation_results.csv sau mỗi seed.
# Nếu bị ngắt giữa chừng, chạy lại nó sẽ tự nhận diện các seed đã xong để bỏ qua.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))

from Common_Data.data_loader import get_train_val_test_data
from CNN_Model.cnn_train import chuan_bi_du_lieu, train_mot_lan
from RandomForest_Baseline.rf_train import encode_label, tao_rf

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEEDS = [42, 43, 44, 45, 46]
CAC_MODE = ["rf", "mlp", "cnn", "hybrid"]
CSV_PATH = BASE_DIR / "ablation_results.csv"
FILE_TAM = BASE_DIR / "CNN_Model" / "tmp_ablation.pth"  # checkpoint tạm, chạy xong xoá


def doc_ket_qua_hien_tai():
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["model", "seed", "test_acc", "test_f1"])


def luu_ket_qua(df):
    df.to_csv(CSV_PATH, index=False)


def main():
    t_bat_dau = time.time()
    df = doc_ket_qua_hien_tai()
    da_chay = set()
    for _, row in df.iterrows():
        da_chay.add((str(row["model"]), int(row["seed"])))

    # ===== 1. Random Forest (chạy trước vì siêu nhanh) =====
    can_chay_rf = [s for s in SEEDS if ("rf", s) not in da_chay]
    if can_chay_rf:
        print("\n[ABLATION] Đang chạy Random Forest cho các seed:", can_chay_rf)
        X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw, _ = get_train_val_test_data()
        X_train = np.log1p(np.maximum(X_train_raw, 0))
        X_test = np.log1p(np.maximum(X_test_raw, 0))
        y_train = encode_label(y_train_raw)
        y_test = encode_label(y_test_raw)

        for seed in can_chay_rf:
            t0 = time.time()
            rf = tao_rf(seed=seed)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            acc = float(accuracy_score(y_test, pred))
            f1 = float(f1_score(y_test, pred, average="macro"))
            print(f"[rf    ] seed={seed} | Test Acc={acc*100:.2f}% | Test F1={f1:.4f} | {time.time()-t0:.1f}s")
            
            new_row = pd.DataFrame([{"model": "rf", "seed": seed, "test_acc": acc, "test_f1": f1}])
            df = pd.concat([df, new_row], ignore_index=True)
            luu_ket_qua(df)
            da_chay.add(("rf", seed))

    # ===== 2. Chuẩn bị dữ liệu cho DL (chỉ tính 1 lần nếu cần) =====
    cac_mode_dl = ["mlp", "cnn", "hybrid"]
    can_chay_dl = any((m, s) not in da_chay for m in cac_mode_dl for s in SEEDS)

    if can_chay_dl:
        data = chuan_bi_du_lieu()

        for mode in cac_mode_dl:
            for seed in SEEDS:
                if (mode, seed) in da_chay:
                    continue
                print(f"\n[ABLATION] Bắt đầu mode={mode.upper()}, seed={seed}...")
                t0 = time.time()
                kq = train_mot_lan(data, mode=mode, seed=seed, save_path=FILE_TAM, in_tung_epoch=False)
                acc = float(kq["test_acc"])
                f1 = float(kq["test_f1"])
                print(f"[{mode:<6}] seed={seed} | best epoch={kq['best_epoch']:>2} | "
                      f"Test Acc={acc*100:.2f}% | Test F1={f1:.4f} | {time.time()-t0:.1f}s")
                
                new_row = pd.DataFrame([{"model": mode, "seed": seed, "test_acc": acc, "test_f1": f1}])
                df = pd.concat([df, new_row], ignore_index=True)
                luu_ket_qua(df)
                da_chay.add((mode, seed))

        if FILE_TAM.exists():
            FILE_TAM.unlink()

    # ===== 3. Tổng hợp bảng kết quả Mean ± Std =====
    print("\n" + "=" * 65)
    print(" BẢNG TỔNG HỢP ABLATION STUDY TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 65)
    print(f"{'Mô hình':<18} {'Số seed':<10} {'Accuracy (%)':<18} {'Macro F1':<18}")
    print("-" * 65)
    for ten in ["rf", "mlp", "cnn", "hybrid"]:
        d = df[df["model"] == ten]
        if len(d) == 0:
            continue
        so_seed = len(d)
        if so_seed > 1:
            acc_str = f"{d['test_acc'].mean()*100:.2f} ± {d['test_acc'].std(ddof=1)*100:.2f}"
            f1_str = f"{d['test_f1'].mean():.4f} ± {d['test_f1'].std(ddof=1):.4f}"
        else:
            acc_str = f"{d['test_acc'].mean()*100:.2f}"
            f1_str = f"{d['test_f1'].mean():.4f}"
        
        ten_hien_thi = {
            "rf": "Random Forest",
            "mlp": "MLP 1D (vector)",
            "cnn": "CNN 2D (ảnh)",
            "hybrid": "Hybrid CNN + MLP",
        }.get(ten, ten)
        print(f"{ten_hien_thi:<18} {so_seed:<10} {acc_str:<18} {f1_str:<18}")

    print("=" * 65)
    print(f"Đã lưu toàn bộ kết quả vào: {CSV_PATH.name}")
    print(f"Tổng thời gian: {(time.time() - t_bat_dau)/60:.1f} phút")


if __name__ == "__main__":
    main()
