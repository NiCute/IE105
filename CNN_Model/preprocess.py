# preprocess.py
# Chuyển dữ liệu dạng bảng (mỗi dòng là 1 vector feature) thành ảnh 24x24 để đưa vào CNN.
# Ý tưởng chính:
#   - gom các feature có ý nghĩa giống nhau vào 1 nhóm (network, crypto, sms,...)
#   - mỗi nhóm bắt đầu ở 1 hàng mới, thiếu thì đệm 0 cho đủ hàng
#     -> kernel 3x3 đỡ bị quét lẫn sang nhóm khác
#   - trong mỗi nhóm thì xếp feature theo mức độ liên quan với nhãn (mutual information, cao xếp trước)

import re

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

# thứ tự các nhóm trên ảnh, từ trên xuống
SEMANTIC_GROUPS_ORDER = [
    "SystemCall",        # 139 syscall của kernel linux
    "Network",           # socket, http, dns...
    "Privacy",           # quyền nhạy cảm, GPS, danh bạ
    "Device",            # phần cứng: sms, phone, wifi, pin, cảm biến
    "Crypto",            # mã hoá, hash
    "Process",           # thread, tạo process
    "Android_Component", # intent, broadcast receiver, service, content provider
    "App_Management",    # cài app, load dex động
    "FileSystem",        # thao tác file
    "Other_API",         # không thuộc nhóm nào thì cho vào đây
]

# keyword để phân nhóm, cái này mình tự ngồi đọc tên cột rồi liệt kê ra
# chắc chắn chưa đầy đủ, nhóm Other_API vẫn còn khá nhiều
SEMANTIC_KEYWORDS = {
    "Network": [
        "NETWORK", "SOCKET", "HTTP", "URL", "INET", "DNS", "CONNECT", "DISCONNECT",
        "BIND", "ACCEPT", "DHCP", "MOBILE", "IFACE", "DATASTATE", "MOBILEDATA", "NET", "TRAFFIC"
    ],
    "Privacy": [
        "PERMISSION", "PERMISSIONS", "ACCOUNT", "ACCOUNTS", "CONTACT", "CONTACTS",
        "LOCATION", "CAMERA", "MIC", "LINE1NUMBER", "ICCSERIAL", "IMEI", "IMSI", 
        "GEOC", "GEOCODER", "PERSONAL", "LATITUDE", "LONGITUDE"
    ],
    "Crypto": [
        "CRYPTO", "CIPHER", "ENCRYPT", "ENCRYPTION", "DECRYPT", "HASH", "SIGN", 
        "SIGNATURE", "SIGNATURES", "KEY", "KEYS", "KEYGUARD", "SSL", "TLS", 
        "SECRET", "AUTHENTICATOR", "CERTIFICATE"
    ],
    "Process": [
        "PROCESS", "THREAD", "EXECUTE", "FORK", "SPAWN", "SCHED", "CLOCK", "EXIT", 
        "KILL", "TERMINATE", "ANTI", "DEBUG", "CLONE", "EXECVE", "CAPSET", "FUTEX", "EVENTFD"
    ],
    "Device": [
        "DEVICE", "PHONE", "SMS", "CALL", "AUDIO", "DISPLAY", "SENSOR", "BATTERY", 
        "GPS", "WIFI", "BLUETOOTH", "VIBRAT", "VIBRATE", "FLASHLIGHT", "CELLINFO", "CELL", "POWER"
    ],
    "Android_Component": [
        "INTENT", "BROADCAST", "RECEIVER", "SERVICE", "ACTIVITY", "CONTENT", 
        "PROVIDER", "REGISTER", "UNREGISTER", "NOTIFICATION", "NOTIFICATIONS", 
        "TOAST", "SYNC", "PANEL", "PANELS", "WINDOW"
    ],
    "App_Management": [
        "PACKAGE", "PACKAGES", "INSTALL", "APP", "APK", "CLASSLOAD", "REFLECT", 
        "PKGUSAGE", "COMPONENT", "COMPONENTS", "DEX", "RESTORE", "ADMIN", "ADMINS"
    ],
    "FileSystem": [
        "FS", "FILE", "FOLDER", "PIPE", "FTRUNCATE", "INOTIFY", "EPOLL", "CHMOD", 
        "MMAP", "SEEK", "FDATASYNC", "FSTAT", "FSYNC", "FLOCK", "FCNTL", "CHDIR", 
        "CHOWN", "ACCESS", "STORAGE"
    ],
}

# trước dùng 22x22 nhưng khi căn hàng theo nhóm thì không đủ chỗ nên tăng lên 24x24
IMAGE_SIZE = (24, 24)
N_COLS = IMAGE_SIZE[1]
N_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 576


def load_syscalls(syscall_file_path):
    # đọc file danh sách syscall, mỗi dòng 1 tên
    if not syscall_file_path.exists():
        return set()
    ds = set()
    with open(syscall_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ds.add(line)
    return ds


class FeatureToImagePipeline:
    # xep_theo: "mi" (mutual information, mặc định) hoặc "pearson" (cách cũ, để lại cho dễ so sánh)
    def __init__(self, feature_names, syscall_file_path, xep_theo="mi"):
        self.feature_names = feature_names
        self.syscall_set = load_syscalls(syscall_file_path)
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.xep_theo = xep_theo
        self.feature_scores = None  # điểm của từng feature, lưu lại để xem sau

        # pixel_mapping[i] = index của feature nằm ở pixel thứ i, -1 nghĩa là ô trống (đệm 0)
        self.pixel_mapping = []
        self.group_layout = {}  # lưu lại nhóm nào nằm ở hàng nào, để debug / vẽ hình

    def classify_feature(self, col_name):
        if col_name in self.syscall_set:
            return "SystemCall"

        # 1. Tách camelCase và PascalCase trước khi chuyển thành chữ hoa
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col_name)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)

        # 2. Tách theo ký tự đặc biệt (_, ., (), `, ...) và đưa về chữ hoa
        tokens = set(re.split(r'[^A-Za-z0-9]+', s.upper()))

        # 3. Đối chiếu từ khóa
        for group, keywords in SEMANTIC_KEYWORDS.items():
            for kw in keywords:
                if kw in tokens:
                    return group
        return "Other_API"

    def fit_transform(self, X, y):
        # chỉ gọi hàm này với tập train thôi, val/test thì gọi transform()
        print("[PREPROCESS] Đang tính toán ma trận ánh xạ 24x24...")

        # B1: chấm điểm mức độ liên quan của từng cột với nhãn
        if self.xep_theo == "pearson":
            # cách cũ: tương quan pearson với nhãn 1..5
            # vấn đề là nhãn là loại mã độc chứ không phải số, pearson coi Benign(5) "lớn hơn" Adware(1)
            # nên không hợp lý lắm -> giờ chuyển qua mutual information
            scores = []
            for i in range(X.shape[1]):
                c = np.corrcoef(X[:, i], y)[0, 1]
                scores.append(c)
            scores = np.abs(scores)
            scores = np.nan_to_num(scores)  # cột toàn 0 thì ra nan
        else:
            # mutual information: đo xem biết giá trị cột này thì đoán nhãn dễ hơn bao nhiêu
            # không quan tâm nhãn có thứ tự hay không, bắt được cả quan hệ phi tuyến
            # log1p trước vì sklearn ước lượng MI bằng kNN, để số đếm cả triệu thì khoảng cách bị lệch
            # random_state cố định vì hàm này có thêm nhiễu ngẫu nhiên vào, không cố định thì mỗi lần chạy ra 1 kiểu
            print("[PREPROCESS] Đang tính mutual information (hơi lâu xíu)...")
            X_log_mi = np.log1p(np.maximum(X, 0))
            scores = mutual_info_classif(X_log_mi, y, random_state=42, n_jobs=-1)
        self.feature_scores = scores

        # B2: chia feature vào các nhóm
        groups = {}
        for g in SEMANTIC_GROUPS_ORDER:
            groups[g] = []
        for idx, name in enumerate(self.feature_names):
            g = self.classify_feature(name)
            groups[g].append((idx, scores[idx]))

        # B3: trong mỗi nhóm sort theo điểm giảm dần
        for g in SEMANTIC_GROUPS_ORDER:
            groups[g].sort(key=lambda item: item[1], reverse=True)

        # B4: xếp vào ảnh, mỗi nhóm bắt đầu từ đầu 1 hàng mới
        self.pixel_mapping = []
        current_row = 0

        for g in SEMANTIC_GROUPS_ORDER:
            features = [item[0] for item in groups[g]]
            count = len(features)
            if count == 0:
                self.group_layout[g] = {"count": 0}
                continue

            self.pixel_mapping.extend(features)

            # số hàng cần = làm tròn lên của count / 24
            rows_needed = (count + N_COLS - 1) // N_COLS
            # phần dư ở hàng cuối thì đệm -1 cho đủ hàng
            pad_count = rows_needed * N_COLS - count
            self.pixel_mapping.extend([-1] * pad_count)

            self.group_layout[g] = {
                "count": count,
                "row_start": current_row,
                "row_end": current_row + rows_needed - 1,
            }
            current_row += rows_needed

        if current_row > IMAGE_SIZE[0]:
            raise ValueError(f"Ảnh không đủ chỗ: cần {current_row} hàng mà chỉ có {IMAGE_SIZE[0]}")

        # còn dư hàng thì đệm nốt cho đủ 576 pixel
        con_lai = N_PIXELS - len(self.pixel_mapping)
        if con_lai > 0:
            self.pixel_mapping.extend([-1] * con_lai)

        # B5: fit scaler
        # data có cột bị gọi tới cả triệu lần nên phải log1p trước, không thì minmax xong gần như toàn 0
        X_log = np.log1p(np.maximum(X, 0))
        self.scaler.fit(X_log)
        self.is_fitted = True

        print(f"[PREPROCESS] Số hàng đã sử dụng: {current_row}/{IMAGE_SIZE[0]}")
        # print(self.group_layout)

        return self.transform(X)

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Chưa fit, gọi fit_transform với tập train trước đã")

        # ảnh có 3 kênh:
        # kênh 1: log1p rồi minmax (fit trên train)
        X_log = np.log1p(np.maximum(X, 0))
        X_norm = self.scaler.transform(X_log)

        # kênh 2: chỉ quan tâm có gọi hay không (0/1)
        # ý tưởng là bắt chước kiểu RF chia nhánh "có/không"
        X_bin = (X > 0).astype(np.float32)

        # kênh 3: chia cho max của chính mẫu đó (normalize theo từng mẫu)
        # cộng 1e-7 để không bị chia cho 0 với mẫu toàn số 0
        X_max_moi_mau = X_log.max(axis=1, keepdims=True)
        X_sample_norm = X_log / (X_max_moi_mau + 1e-7)

        N = len(X)
        c1 = np.zeros((N, N_PIXELS), dtype=np.float32)
        c2 = np.zeros((N, N_PIXELS), dtype=np.float32)
        c3 = np.zeros((N, N_PIXELS), dtype=np.float32)

        for px_idx, feat_idx in enumerate(self.pixel_mapping):
            if feat_idx == -1:
                continue  # ô trống, để nguyên 0
            c1[:, px_idx] = X_norm[:, feat_idx]
            c2[:, px_idx] = X_bin[:, feat_idx]
            c3[:, px_idx] = X_sample_norm[:, feat_idx]

        # reshape về (N, 1, 24, 24) rồi ghép 3 kênh lại -> (N, 3, 24, 24)
        c1 = c1.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        c2 = c2.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        c3 = c3.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])

        images = np.concatenate([c1, c2, c3], axis=1)
        return images
