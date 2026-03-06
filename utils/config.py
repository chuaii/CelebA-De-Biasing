import os

ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, "datasets")
IMG_DIR = os.path.join(DATA_ROOT, "img_align_celeba")
ATTR_CSV = os.path.join(DATA_ROOT, "list_attr_celeba.csv")
PARTITION_CSV = os.path.join(DATA_ROOT, "list_eval_partition.csv")

TARGET_ATTR = "Blond_Hair"
SENSITIVE_ATTR = "Male"

# group = target * 2 + sensitive
GROUP_NAMES = {0: "NonBlond_Female", 1: "NonBlond_Male", 2: "Blond_Female", 3: "Blond_Male"}

BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_EPOCHS = 20
LR = 1e-4
LR_BACKBONE = 1e-5
WD = 1e-4
EMBED_DIM = 128
TEMPERATURE = 0.07
LAMBDA_CON = 0.2  # 0 = baseline, >0 = debias

DEVICE = "cuda"
SEED = 42
CKPT_DIR = os.path.join(ROOT, "checkpoints")
