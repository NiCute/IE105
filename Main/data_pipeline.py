"""
data_pipeline.py
================
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Đường ống tiền xử lý dữ liệu và kỹ thuật trích chọn đặc trưng hành vi (Feature Engineering Pipeline).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Danh sách từ khóa để gom nhóm 9 nhóm ngữ nghĩa hành vi
SEMANTIC_KEYWORDS = {
    "FileSystem": [
        "FS_ACCESS", "FS_PIPE", "CREATE_FOLDER", "FTRUNCATE",
        "INOTIFY", "EPOLL", "CHMOD", "MMAP", "_LLSEEK", "FDATASYNC",
        "FSTAT", "FSTATFS", "FSYNC", "FLOCK", "FCNTL", "CHDIR",
        "CHOWN", "FACCESSAT", "FCHOWN", "ACCESS",
    ],
    "Network": [
        "NETWORK", "SOCKET", "HTTP", "URL", "INET", "DNS",
        "CONNECT", "BIND", "ACCEPT", "DHCP", "MOBILE_IFACE",
        "GETDATASTATE", "GETMOBILEDATA",
    ],
    "Privacy": [
        "PERMISSION", "ACCESS_PERSONAL", "ACCOUNT", "CONTACT",
        "LOCATION", "CAMERA", "MIC", "LINE1NUMBER", "ICCSERIAL",
        "IMEI", "IMSI", "GEOC",
    ],
    "Crypto": [
        "CRYPTO", "CIPHER", "ENCRYPT", "HASH", "SIGN",
        "KEY", "SSL", "TLS", "SECRET",
    ],
    "Process": [
        "PROCESS", "THREAD", "EXECUTE", "FORK", "SPAWN",
        "SCHED", "CLOCK", "EXIT", "KILL", "TERMINATE",
        "ANTI_DEBUG", "CLONE", "EXECVE", "CAPSET", "FUTEX",
        "EVENTFD", "CREATE_THREAD", "CREATE_PROCESS",
    ],
    "Device": [
        "DEVICE", "PHONE", "SMS", "CALL", "AUDIO", "DISPLAY",
        "SENSOR", "BATTERY", "GPS", "WIFI", "BLUETOOTH",
        "VIBRAT", "FLASHLIGHT", "CELLINFO", "NEIGHBORINGCELL",
        "ALTER_PHONE",
    ],
    "Android_Component": [
        "INTENT", "BROADCAST", "RECEIVER", "SERVICE", "ACTIVITY",
        "CONTENT", "PROVIDER", "REGISTER", "UNREGIST",
        "NOTIFICATION", "TOAST", "SYNC", "PANEL",
    ],
    "App_Management": [
        "PACKAGE", "INSTALL", "APP", "APK", "CLASSLOAD",
        "REFLECT", "PKGUSAGE", "COMPONENT_ENABLED",
        "GETCOMPONENT",
    ],
}


class FeaturePipeline:
    """
    Quy trình tiền xử lý và tạo đặc trưng:
    1. Lọc bỏ các cột trùng lặp giá trị 100%.
    2. Căn chỉnh thứ tự cột đặc trưng cố định.
    3. Tạo các đặc trưng hành vi tổng hợp (Feature Engineering):
       - feat_total_calls: Tổng số lần gọi API của mỗi mẫu.
       - feat_unique_calls: Số lượng API khác nhau được kích hoạt.
       - grp_<Tên_Nhóm>_sum & grp_<Tên_Nhóm>_unique: Tần suất và số lượng theo từng họ hành vi.
       - feat_risk_ratio: Tỷ lệ các API nhạy cảm (Privacy + Device) trên tổng hoạt động.
    4. Áp dụng Log-transform (np.log1p) triệt tiêu ngoại lai.
    """

    def __init__(self):
        self.raw_feature_cols = None
        self.non_redundant_cols = None
        self.final_feature_cols = None
        self.group_cols = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        """Học cấu trúc cột trên tập Huấn luyện (Train) để tránh Data Leakage."""
        self.raw_feature_cols = list(X.columns)

        # 1. Phát hiện 14 cột trùng lặp hoàn toàn
        dup_mask = X.T.duplicated()
        self.non_redundant_cols = [col for col, is_dup in zip(X.columns, dup_mask) if not is_dup]

        # 2. Phân loại các cột vào từng nhóm ngữ nghĩa
        for group, kws in SEMANTIC_KEYWORDS.items():
            cols = [c for c in self.non_redundant_cols if any(kw in c.upper() for kw in kws)]
            self.group_cols[group] = cols

        # 3. Tạo mẫu thử để xác định danh sách các cột sau cùng
        sample_transformed = self._engineer_features(X[self.non_redundant_cols])
        self.final_feature_cols = list(sample_transformed.columns)
        self.is_fitted = True
        return self

    def _engineer_features(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        """Tạo các đặc trưng hành vi tổng hợp."""
        feat_df = df_sub.copy()

        # Tạo thống kê theo từng nhóm ngữ nghĩa
        for group, cols in self.group_cols.items():
            if cols:
                feat_df[f"grp_{group}_sum"] = df_sub[cols].sum(axis=1)
                feat_df[f"grp_{group}_unique"] = (df_sub[cols] > 0).sum(axis=1)

        # Tổng hoạt động toàn cục
        feat_df["feat_total_calls"] = df_sub.sum(axis=1)
        feat_df["feat_unique_calls"] = (df_sub > 0).sum(axis=1)

        # Tỷ lệ hành vi nguy hiểm (Privacy + Device)
        privacy_sum = feat_df.get("grp_Privacy_sum", 0)
        device_sum = feat_df.get("grp_Device_sum", 0)
        feat_df["feat_risk_ratio"] = (privacy_sum + device_sum) / (feat_df["feat_total_calls"] + 1.0)

        return feat_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Biến đổi dữ liệu mới dựa trên các thuộc tính đã học được khi fit."""
        if not self.is_fitted:
            raise RuntimeError("Cần gọi fit() trước khi transform()!")

        # Căn chỉnh các cột bằng dictionary để tối ưu hiệu năng và tránh phân mảnh DataFrame
        aligned_dict = {}
        for col in self.non_redundant_cols:
            if col in X.columns:
                aligned_dict[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).values
            else:
                aligned_dict[col] = np.zeros(len(X), dtype=np.float32)
        aligned_df = pd.DataFrame(aligned_dict, index=X.index)

        # Tạo đặc trưng mới
        engineered = self._engineer_features(aligned_df)

        # Khớp chính xác thứ tự cột
        engineered = engineered[self.final_feature_cols]

        # Áp dụng Log-transform np.log1p
        X_log = np.log1p(np.maximum(engineered.values, 0))
        return pd.DataFrame(X_log, columns=self.final_feature_cols, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, filepath: Path):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "FeaturePipeline":
        return joblib.load(filepath)
