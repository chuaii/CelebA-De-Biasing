"""
LfF (Learning from Failure) 方法配置文件。
复用主项目的数据路径与群组定义，定义 LfF 专属超参数。
"""
import config

# ---- 从主配置复用的路径与定义 ----
ROOT = config.ROOT
DATA_ROOT = config.DATA_ROOT
IMG_DIR = config.IMG_DIR
ATTR_CSV = config.ATTR_CSV
PARTITION_CSV = config.PARTITION_CSV
TARGET_ATTR = config.TARGET_ATTR
SENSITIVE_ATTR = config.SENSITIVE_ATTR
GROUP_NAMES = config.GROUP_NAMES
CKPT_DIR = config.CKPT_DIR
SEED = config.SEED
DEVICE = config.DEVICE
NUM_WORKERS = config.NUM_WORKERS

# ---- Stage 1: Biased Model (有偏模型) ----
BIASED_EPOCHS = 5          # 少量 epoch，使模型仅学到偏见捷径
BIASED_LR = 1e-3           # 较大学习率，快速收敛到偏见模式
BIASED_BATCH_SIZE = 128
GCE_Q = 0.7                # Generalized CE 截断参数（越大越聚焦于"简单"样本）

# ---- Stage 2: Weight Normalization (权重归一化) ----
WEIGHT_CLAMP_MIN = 0.1     # 最小样本权重，避免完全忽略简单样本
WEIGHT_CLAMP_MAX = 10.0    # 最大样本权重，避免不稳定

# ---- Stage 3: Main / Debiased Model (去偏主模型) ----
MAIN_EPOCHS = 20
MAIN_LR = 1e-4
MAIN_LR_BACKBONE = 1e-5    # backbone 用更低学习率保留预训练特征
MAIN_BATCH_SIZE = 128
MAIN_WD = 1e-4
MAIN_PATIENCE = 7           # early stopping 容忍轮数
