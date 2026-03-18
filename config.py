import os

ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, "datasets")
IMG_DIR = os.path.join(DATA_ROOT, "img_align_celeba")
ATTR_CSV = os.path.join(DATA_ROOT, "list_attr_celeba.csv")
PARTITION_CSV = os.path.join(DATA_ROOT, "list_eval_partition.csv")

# group = target * 2 + sensitive
# TARGET_ATTR = "Blond_Hair"
# SENSITIVE_ATTR = "Male"
# GROUP_NAMES = {0: "NonBlond_Female", 1: "NonBlond_Male", 2: "Blond_Female", 3: "Blond_Male"}

# Mouth_Slightly_Open × Smiling 
# TARGET_ATTR = "Mouth_Slightly_Open"
# SENSITIVE_ATTR = "Smiling"
# GROUP_NAMES = {0: "MouthNonOpen_NonSmiling", 1: "MouthNonOpen_Smiling", 2: "MouthOpen_NonSmiling", 3: "MouthOpen_Smiling"}

# Smiling / High_Cheekbones
TARGET_ATTR = "Smiling"
SENSITIVE_ATTR = "High_Cheekbones"
GROUP_NAMES = {0: "NonSmiling_HighCheekbones", 1: "NonSmiling_LowCheekbones", 2: "Smiling_HighCheekbones", 3: "Smiling_LowCheekbones"}

# # Young × Gray_Hair
# TARGET_ATTR = "Young"
# SENSITIVE_ATTR = "Gray_Hair"
# GROUP_NAMES = {0: "Young_NonGray", 1: "Young_Gray", 2: "NonYoung_NonGray", 3: "NonYoung_Gray"}


BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1  # 前 N 个 epoch 线性 warmup，0 表示不做 warmup
LR = 1e-5
LR_BACKBONE = 1e-6
WD = 1e-4
EMBED_DIM = 128
TEMPERATURE = 0.07 # SupCon default: 0.07
LAMBDA_CON = 1.5  # 0 = baseline, >0 = debias

DEVICE = "cuda"
SEED = 42
CKPT_DIR = os.path.join(ROOT, "checkpoints")
