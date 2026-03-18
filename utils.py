import os
import random
import numpy as np
import torch
import config as cfg


def set_seed(s=cfg.SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")


def log_epoch(ep, total_epochs, loss, metrics, extra=""):
    line = (f"[{ep+1}/{total_epochs}] loss={loss:.4f} acc={metrics['overall_acc']:.2%}  wga={metrics['worst_group_acc']:.2%}")
    if "eqodd" in metrics:
        line += f"  eqodd={metrics['eqodd']:.4f}"
    if extra:
        line += f"  {extra}"
    print(line)
    for g in range(4):
        print(f"  {cfg.GROUP_NAMES[g]}: {metrics['group_acc'][g]:.2%}")


class BestTracker:
    """按 WGA / EqOdd 维护 best checkpoint。warmup 期间不保存。"""

    def __init__(self, tag, warmup_epochs=0):
        self.tag = tag
        self.warmup_epochs = warmup_epochs
        self.best_wga = 0.0
        self.best_eqodd = float("inf")

    def _save(self, model, suffix):
        os.makedirs(cfg.CKPT_DIR, exist_ok=True)
        path = os.path.join(cfg.CKPT_DIR, f"best_{self.tag}_{suffix}.pt")
        torch.save(model.state_dict(), path)
        return path

    def update(self, model, metrics, epoch=0):
        """根据 metrics dict（含 worst_group_acc / eqodd）更新 best。
        warmup 期间跳过保存。
        """
        if epoch < self.warmup_epochs:
            return
        wga = metrics["worst_group_acc"]
        eqodd = metrics["eqodd"]
        saved = []

        if wga > self.best_wga:
            self.best_wga = wga
            saved.append(self._save(model, "wga"))
        if eqodd < self.best_eqodd:
            self.best_eqodd = eqodd
            saved.append(self._save(model, "eqodd"))
        if saved:
            print(f"  -> saved {', '.join(os.path.basename(p) for p in saved)}")

    def summary(self):
        return f"best wga={self.best_wga:.2%}  best eqodd={self.best_eqodd:.4f}"
