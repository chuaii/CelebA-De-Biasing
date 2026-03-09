import os
from collections import Counter
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import config as cfg

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CelebAFairness(Dataset):
    """返回 (image, target, sensitive, group)。"""

    def __init__(self, split="train", transform=None):
        attr = pd.read_csv(cfg.ATTR_CSV)
        part = pd.read_csv(cfg.PARTITION_CSV)
        df = attr.merge(part, on="image_id")
        df = df[df["partition"] == {"train": 0, "val": 1, "test": 2}[split]].reset_index(drop=True)

        self.filenames = df["image_id"].tolist()
        self.targets = ((df[cfg.TARGET_ATTR] + 1) // 2).astype(int).tolist()
        self.sensitives = ((df[cfg.SENSITIVE_ATTR] + 1) // 2).astype(int).tolist()
        self.groups = [t * 2 + s for t, s in zip(self.targets, self.sensitives)]
        self.transform = transform

        cnt = Counter(self.groups)
        stats = " | ".join(f"{cfg.GROUP_NAMES[g]}={cnt[g]}" for g in sorted(cnt))
        print(f"[{split}] n={len(self)}  {stats}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(cfg.IMG_DIR, self.filenames[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx], self.sensitives[idx], self.groups[idx]


def _group_balanced_sampler(dataset):
    """每个样本的采样权重 = 1/该组样本数，使各组在期望上被均匀采到。"""
    cnt = Counter(dataset.groups)
    weights = [1.0 / cnt[g] for g in dataset.groups]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def get_loader(split, batch_size=None, balanced=False):
    is_train = split == "train"
    ds = CelebAFairness(split, train_transform if is_train else eval_transform)
    sampler = _group_balanced_sampler(ds) if (is_train and balanced) else None
    return DataLoader(
        ds, 
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True, 
        drop_last=is_train
        )
