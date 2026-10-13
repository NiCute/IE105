import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.resolve()

# Lúc trước code đọc file "139_extracted features.csv" trong thư mục dataset,
# mà file đó chỉ có 139 cột syscall -> phần gom nhóm bên preprocess.py gần như vô dụng
# (cả 139 cột đều rơi vào nhóm SystemCall), README thì lại ghi 470 đặc trưng.
# => thống nhất dùng file 471 cột (470 feature + 1 cột Class). File 471 có chứa luôn 139 cột kia.
DATA_PATH = BASE_DIR / "471_extracted features.csv"


# hàm này dùng chung cho cả CNN và RF, 2 model chia data giống nhau thì so sánh mới công bằng
# data_path để mặc định là file 471, truyền file khác vào nếu muốn thử
def get_train_val_test_data(data_path=DATA_PATH):
    data_path = Path(data_path)
    print(f"\n[DỮ LIỆU] Đang nạp tệp: {data_path.name}")
    df = pd.read_csv(data_path)
    so_dong_ban_dau = len(df)

    # bỏ mấy dòng trùng y chang nhau
    df = df.drop_duplicates()

    # có một số dòng features giống hệt nhau mà label lại khác -> bỏ luôn cả 2
    # (lúc đầu chỉ drop_duplicates thôi nhưng vẫn còn bị, check kỹ mới thấy)
    feature_cols = df.columns[:-1].tolist()
    bi_xung_dot = df.duplicated(subset=feature_cols, keep=False)
    df = df[~bi_xung_dot].reset_index(drop=True)

    so_dong_bo = so_dong_ban_dau - len(df)
    if so_dong_bo > 0:
        print(f"  Đã loại bỏ {so_dong_bo} mẫu trùng lặp/xung đột -> Còn lại: {len(df)} mẫu")

    label_col = df.columns[-1]  # cột cuối là nhãn (Class)

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    # chia 70/15/15
    # tách 15% test ra trước, phần còn lại (85%) chia tiếp thành train với val
    # để 0.15/0.85 là để val ra đúng 15% của tổng
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )

    print(f"  Số đặc trưng: {len(feature_cols)}")
    print("\n[PHÂN CHIA TẬP DỮ LIỆU 70 / 15 / 15]")
    print(f"  - Train : {len(X_train)} mẫu")
    print(f"  - Val   : {len(X_val)} mẫu")
    print(f"  - Test  : {len(X_test)} mẫu")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
