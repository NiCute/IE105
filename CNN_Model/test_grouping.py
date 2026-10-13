# file test nhanh: xem mỗi nhóm có bao nhiêu feature
# và nếu căn hàng thì ảnh 24x24 có đủ chỗ không
# (viết tạm để kiểm tra trước khi sửa preprocess.py, giờ vẫn để lại để check khi đổi keyword)

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from CNN_Model.preprocess import FeatureToImagePipeline, SEMANTIC_GROUPS_ORDER

BASE_DIR = Path(__file__).parent.resolve()
SYSCALL_FILE = BASE_DIR.parent / "Common_Data" / "syscall_unique.csv"
DATA_FILE = BASE_DIR.parent / "Common_Data" / "471_extracted features.csv"

df = pd.read_csv(DATA_FILE)
feature_cols = df.columns[:-1].tolist()

# dùng lại hàm phân nhóm bên preprocess luôn cho khỏi copy lại keyword
p = FeatureToImagePipeline(feature_cols, SYSCALL_FILE)

groups = {}
for g in SEMANTIC_GROUPS_ORDER:
    groups[g] = []
for name in feature_cols:
    groups[p.classify_feature(name)].append(name)

for g in SEMANTIC_GROUPS_ORDER:
    print(f"{g}: {len(groups[g])} features")

# đếm số hàng cần nếu mỗi nhóm bắt đầu ở 1 hàng mới
total_rows = 0
for g in SEMANTIC_GROUPS_ORDER:
    count = len(groups[g])
    if count == 0:
        continue
    total_rows += (count + 23) // 24  # làm tròn lên

print(f"Total rows needed: {total_rows} / 24")  # in tieng Viet ra cmd bi loi font nen de tieng Anh
