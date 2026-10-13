# cnn_train.py
# Train model CNN phân loại mã độc Android từ ảnh 24x24 (tạo ở preprocess.py)
#
# Bản đầu chỉ có CNN thường thì được ~86%, sau đó thử thêm dần mấy cái này:
#   - ảnh 3 kênh (log-minmax, nhị phân, chuẩn hoá theo mẫu)
#   - bỏ Global Average Pooling, thay bằng AdaptiveAvgPool 3x3 + CoordConv để không mất thông tin vị trí
#   - thêm 1 nhánh MLP chạy song song với CNN, nhận vector gốc
#   - lưu model theo macro F1 thay vì accuracy (do lớp Adware ít mẫu, acc cao mà Adware vẫn sai nhiều)
#   - label smoothing 0.1 + OneCycleLR
#
# Cách chạy:
#   python CNN_Model/cnn_train.py                    -> train bản hybrid (mặc định), lưu model cho predict_cnn.py
#   python CNN_Model/cnn_train.py --mode cnn         -> chỉ nhánh CNN
#   python CNN_Model/cnn_train.py --mode mlp         -> chỉ nhánh MLP
#   (muốn chạy hết 3 kiểu x 5 seed thì dùng run_ablation.py)

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

# thêm thư mục cha vào path để import được Common_Data
sys.path.append(str(Path(__file__).parent.parent))
from Common_Data.data_loader import get_train_val_test_data
from CNN_Model.preprocess import FeatureToImagePipeline

# trên windows print tiếng Việt hay bị lỗi encoding nên thêm đoạn này
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

N_CLASSES = 5
CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent.resolve()

# mấy tham số train để chung 1 chỗ cho dễ sửa
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 12


def set_seed(seed=42):
    # cố định seed để mỗi lần chạy ra kết quả giống nhau
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AddCoords(nn.Module):
    # CoordConv: ghép thêm 2 kênh toạ độ x, y vào ảnh
    # CNN bình thường không biết pixel đang ở vị trí nào, mà ở bài này vị trí lại quan trọng
    # (mỗi hàng là 1 nhóm feature khác nhau) nên thêm vào cho model biết
    def __init__(self):
        super().__init__()

    def forward(self, x_img):
        b, c, h, w = x_img.shape
        xx = torch.arange(w, dtype=x_img.dtype, device=x_img.device).view(1, 1, 1, w).expand(b, 1, h, w)
        yy = torch.arange(h, dtype=x_img.dtype, device=x_img.device).view(1, 1, h, 1).expand(b, 1, h, w)
        # đưa về khoảng [-1, 1]
        xx = xx / (w - 1) * 2 - 1
        yy = yy / (h - 1) * 2 - 1
        return torch.cat([x_img, xx, yy], dim=1)


class HybridMalwareModel(nn.Module):
    # mode:
    #   "hybrid" : CNN + MLP (bản chính)
    #   "cnn"    : chỉ nhánh CNN   } 2 cái này để làm ablation,
    #   "mlp"    : chỉ nhánh MLP   } xem mỗi nhánh đóng góp bao nhiêu
    def __init__(self, n_features, n_classes=N_CLASSES, dropout=0.25, dropout_fc=0.5, mode="hybrid"):
        super().__init__()
        if mode not in ["hybrid", "cnn", "mlp"]:
            raise ValueError(f"mode không hợp lệ: {mode}")
        self.mode = mode
        dau_vao_classifier = 0

        # ----- nhánh CNN -----
        if mode in ["hybrid", "cnn"]:
            self.coord_add = AddCoords()
            # input: 3 kênh ảnh + 2 kênh toạ độ = 5
            self.cnn_block1 = nn.Sequential(
                nn.Conv2d(5, 32, kernel_size=3, padding=0),   # 24 -> 22
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.cnn_block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=0),  # 22 -> 20
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),        # 20 -> 10
                nn.Dropout2d(dropout),
            )
            self.cnn_block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1), # giữ 10x10
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            )
            # hồi trước dùng GAP (pool về 1x1) thì mất hết thông tin vị trí, acc tụt
            # đổi sang pool về 3x3 thì ổn hơn
            self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
            self.cnn_flatten = nn.Flatten()
            dau_vao_classifier += 128 * 3 * 3  # = 1152

        # ----- nhánh MLP -----
        # nhận thẳng vector feature gốc (đã log1p)
        if mode in ["hybrid", "mlp"]:
            self.mlp_branch = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
            dau_vao_classifier += 128

        # ----- ghép 2 nhánh rồi phân loại -----
        # 3 mode đều dùng chung classifier này, chỉ khác số chiều đầu vào
        self.classifier = nn.Sequential(
            nn.Linear(dau_vao_classifier, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(512, n_classes),
        )

    def forward(self, x_img, x_raw):
        cac_dac_trung = []

        if self.mode in ["hybrid", "cnn"]:
            x = self.coord_add(x_img)
            x = self.cnn_block1(x)
            x = self.cnn_block2(x)
            x = self.cnn_block3(x)
            x = self.spatial_pool(x)
            cac_dac_trung.append(self.cnn_flatten(x))

        if self.mode in ["hybrid", "mlp"]:
            cac_dac_trung.append(self.mlp_branch(x_raw))

        out = torch.cat(cac_dac_trung, dim=1)
        return self.classifier(out)


def encode_label(y):
    # nhãn trong file csv là 1..5, đổi về 0..4 cho CrossEntropyLoss
    # map cứng luôn chứ không dùng LabelEncoder, lỡ tập nào thiếu 1 lớp thì bị lệch index
    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    encoded = [label_map[int(v)] for v in y]
    return torch.tensor(encoded, dtype=torch.long)


class EarlyStopping:
    # dừng sớm nếu sau `patience` epoch mà điểm val không tăng
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float("inf")
        self.should_stop = False

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x_img, x_raw, y in loader:
        x_img = x_img.to(device)
        x_raw = x_raw.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_img, x_raw)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # OneCycleLR thì step theo từng batch chứ không phải theo epoch

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for x_img, x_raw, y in loader:
        x_img = x_img.to(device)
        x_raw = x_raw.to(device)
        y = y.to(device)

        logits = model(x_img, x_raw)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / total, correct / total, macro_f1, all_preds, all_labels


def chuan_bi_du_lieu():
    # load data + tạo ảnh, tách ra hàm riêng để run_ablation.py gọi 1 lần rồi dùng lại cho nhiều lần train
    # (tính mutual information khá lâu, mỗi lần train lại tính lại thì phí)
    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw, feature_cols = get_train_val_test_data()

    # input cho nhánh MLP: log1p cho bớt lệch
    X_train_log = np.log1p(np.maximum(X_train_raw, 0))
    X_val_log = np.log1p(np.maximum(X_val_raw, 0))
    X_test_log = np.log1p(np.maximum(X_test_raw, 0))

    # đổi sang ảnh cho nhánh CNN
    syscall_path = BASE_DIR.parent / "Common_Data" / "syscall_unique.csv"
    pipeline = FeatureToImagePipeline(feature_cols, syscall_path)

    X_train_img = pipeline.fit_transform(X_train_raw, y_train_raw)  # fit trên train thôi
    X_val_img = pipeline.transform(X_val_raw)
    X_test_img = pipeline.transform(X_test_raw)

    y_train = encode_label(y_train_raw)
    y_val = encode_label(y_val_raw)
    y_test = encode_label(y_test_raw)

    # cho hết vào dict cho gọn
    data = {
        "n_features": len(feature_cols),
        "pipeline": pipeline,
        "train": TensorDataset(torch.tensor(X_train_img), torch.tensor(X_train_log, dtype=torch.float32), y_train),
        "val": TensorDataset(torch.tensor(X_val_img), torch.tensor(X_val_log, dtype=torch.float32), y_val),
        "test": TensorDataset(torch.tensor(X_test_img), torch.tensor(X_test_log, dtype=torch.float32), y_test),
        "y_train": y_train,
    }
    return data


def train_mot_lan(data, mode="hybrid", seed=42, save_path=None, in_tung_epoch=True):
    # train 1 model từ đầu tới cuối rồi trả về kết quả trên tập test
    # set seed ngay đầu hàm để init trọng số + thứ tự shuffle của mỗi lần chạy là cố định
    set_seed(seed)

    if save_path is None:
        save_path = BASE_DIR / f"model_{mode}.pth"

    train_loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data["val"], batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(data["test"], batch_size=BATCH_SIZE * 2, shuffle=False)

    # data bị lệch lớp nên dùng class weight
    # để nguyên weight 'balanced' thì model phạt lớp ít mẫu quá tay, lấy căn bậc 2 cho nhẹ bớt
    y_train = data["y_train"]
    unique_lbl, counts = torch.unique(y_train, return_counts=True)
    class_weights = len(y_train) / (len(unique_lbl) * counts.float())
    class_weights = torch.sqrt(class_weights)

    model = HybridMalwareModel(n_features=data["n_features"], n_classes=N_CLASSES, mode=mode).to(DEVICE)

    # label_smoothing để model đỡ "tự tin thái quá"
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # OneCycleLR: lr tăng dần lên max_lr rồi giảm dần về cuối
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    stopper = EarlyStopping(patience=PATIENCE)
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        # lưu model khi macro F1 trên val tốt hơn
        note = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "class_weights": class_weights,
                "mode": mode,
            }, save_path)
            note = "  << BEST F1"

        if in_tung_epoch:
            print(f"Epoch {epoch:>3}/{EPOCHS} | Train Loss={tr_loss:.4f} Acc={tr_acc*100:.2f}% | "
                  f"Val Loss={va_loss:.4f} Acc={va_acc*100:.2f}% F1={va_f1:.4f}{note}")

        stopper.step(va_f1)
        if stopper.should_stop:
            if in_tung_epoch:
                print(f"Dừng sớm ở epoch {epoch} vì F1 không tăng nữa")
            break

    # load lại model tốt nhất rồi mới test
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc, te_f1, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)

    ket_qua = {
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_f1": best_f1,
        "test_acc": accuracy_score(all_labels, all_preds),
        "test_f1": te_f1,
        "preds": all_preds,
        "labels": all_labels,
    }
    return ket_qua


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "cnn", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Thiết bị: {DEVICE} | mode: {args.mode} | seed: {args.seed}")
    print("=" * 60)
    t0 = time.time()

    data = chuan_bi_du_lieu()

    # chỉ bản hybrid mới lưu ra cnn_best_model.pth + pipeline (predict_cnn.py dùng 2 file này)
    # mấy mode khác lưu tên khác để không bị ghi đè lên model chính
    if args.mode == "hybrid":
        save_path = BASE_DIR / "cnn_best_model.pth"
        joblib.dump(data["pipeline"], BASE_DIR / "cnn_pipeline.joblib")
    else:
        save_path = BASE_DIR / f"model_{args.mode}.pth"

    print(f"\n[HUẤN LUYỆN - {args.mode.upper()}] Bắt đầu...")
    kq = train_mot_lan(data, mode=args.mode, seed=args.seed, save_path=save_path)

    print(f"\nTrain xong, mất {time.time() - t0:.1f}s (model tốt nhất ở epoch {kq['best_epoch']})")

    print("\n[ĐÁNH GIÁ TRÊN TẬP TEST]")
    print(f"Test Accuracy: {kq['test_acc']*100:.4f}%")
    print(f"Test Macro F1: {kq['test_f1']:.4f}")
    print(classification_report(kq["labels"], kq["preds"], target_names=CLASS_NAMES, digits=4))


if __name__ == "__main__":
    main()
