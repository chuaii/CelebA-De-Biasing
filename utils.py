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
    if "eod" in metrics:
        line += f"  eod={metrics['eod']:.4f}  eqodd={metrics['eqodd']:.4f}  dpd={metrics['dpd']:.4f}"
    if extra:
        line += f"  {extra}"
    print(line)
    for g in range(4):
        print(f"  {cfg.GROUP_NAMES[g]}: {metrics['group_acc'][g]:.2%}")


# def save_best(model, wga, best_wga, tag):
#     """Save checkpoint when wga improves. Returns updated best_wga."""
#     if wga > best_wga:
#         os.makedirs(cfg.CKPT_DIR, exist_ok=True)
#         path = os.path.join(cfg.CKPT_DIR, f"best_{tag}.pt")
#         torch.save(model.state_dict(), path)
#         print(f"  -> saved {path}")
#         return wga
#     return best_wga


class BestTracker:
    """同时按 WGA / EOD / EqOdd / tradeoff 维护多份 best checkpoint。
    tradeoff 规则：WGA 距最佳不超过 delta 时，选 EOD 最小的。
    """

    def __init__(self, tag, delta=0.005):
        self.tag = tag
        self.delta = delta
        self.best_wga = 0.0
        self.best_eod = float("inf")
        self.best_eqodd = float("inf")
        self.tradeoff_wga = 0.0
        self.tradeoff_eod = float("inf")

    def _save(self, model, suffix):
        os.makedirs(cfg.CKPT_DIR, exist_ok=True)
        path = os.path.join(cfg.CKPT_DIR, f"best_{self.tag}_{suffix}.pt")
        torch.save(model.state_dict(), path)
        return path

    def update(self, model, metrics):
        """根据 metrics dict（含 worst_group_acc / eod / eqodd）更新所有 best。"""
        wga = metrics["worst_group_acc"]
        eod = metrics["eod"]
        eqodd = metrics["eqodd"]
        saved = []

        if wga > self.best_wga:
            self.best_wga = wga
            saved.append(self._save(model, "wga"))

        if eod < self.best_eod:
            self.best_eod = eod
            saved.append(self._save(model, "eod"))

        if eqodd < self.best_eqodd:
            self.best_eqodd = eqodd
            saved.append(self._save(model, "eqodd"))

        if wga >= self.best_wga - self.delta and eod < self.tradeoff_eod:
            self.tradeoff_wga = wga
            self.tradeoff_eod = eod
            saved.append(self._save(model, "tradeoff"))

        if saved:
            print(f"  -> saved {', '.join(os.path.basename(p) for p in saved)}")

    def summary(self):
        return (f"best wga={self.best_wga:.2%}  "
                f"best eod={self.best_eod:.4f}  "
                f"best eqodd={self.best_eqodd:.4f}  "
                f"tradeoff wga={self.tradeoff_wga:.2%}/eod={self.tradeoff_eod:.4f}")
