"""
LfF 数据加载工具。

在原始 CelebAFairness 的基础上新增索引返回，
用于 Stage 2 的逐样本损失计算和 Stage 3 的逐样本权重查找。
"""
from torch.utils.data import Dataset, DataLoader
from dataset import CelebAFairness, train_transform, eval_transform
import LfF_config as cfg


class IndexedCelebA(Dataset):
    """在原始 CelebAFairness 返回值中追加样本索引 idx。"""

    def __init__(self, split="train", transform=None):
        self.inner = CelebAFairness(split, transform)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, target, sensitive, group = self.inner[idx]
        return img, target, sensitive, group, idx

    @property
    def groups(self):
        return self.inner.groups

    @property
    def targets(self):
        return self.inner.targets

    @property
    def sensitives(self):
        return self.inner.sensitives


def get_train_loader(batch_size=None):
    """训练集 DataLoader（带索引、随机增强、shuffle），用于 Stage 1 和 Stage 3。"""
    bs = batch_size or cfg.BIASED_BATCH_SIZE
    ds = IndexedCelebA("train", train_transform)
    return DataLoader(
        ds, batch_size=bs, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
    )


def get_weight_loader(batch_size=None):
    """
    权重计算专用 DataLoader（带索引、eval 变换、顺序遍历、不丢弃末尾）。
    使用 eval 变换（无随机增强）以获得确定性的逐样本损失值。
    """
    bs = batch_size or cfg.BIASED_BATCH_SIZE
    ds = IndexedCelebA("train", eval_transform)
    return DataLoader(
        ds, batch_size=bs, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
    )


def get_eval_loader(split, batch_size=None):
    """验证/测试集 DataLoader。"""
    bs = batch_size or cfg.MAIN_BATCH_SIZE
    ds = CelebAFairness(split, eval_transform)
    return DataLoader(
        ds, batch_size=bs, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
    )
