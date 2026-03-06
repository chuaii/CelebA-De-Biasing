import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import config


def set_seed(s=config.SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def log_epoch(ep, total_epochs, elapsed, loss, metrics, extra=""):
    line = (f"[{ep+1}/{total_epochs}] {elapsed:.0f}s  loss={loss:.4f}  "
            f"acc={metrics['overall_acc']:.2%}  wga={metrics['worst_group_acc']:.2%}")
    if extra:
        line += f"  {extra}"
    print(line)
    for g in range(4):
        print(f"  {config.GROUP_NAMES[g]}: {metrics['group_acc'][g]:.2%}")


def save_best(model, wga, best_wga, tag):
    """Save checkpoint when wga improves. Returns updated best_wga."""
    if wga > best_wga:
        os.makedirs(config.CKPT_DIR, exist_ok=True)
        path = os.path.join(config.CKPT_DIR, f"best_{tag}.pt")
        torch.save(model.state_dict(), path)
        print(f"  -> saved {path}")
        return wga
    return best_wga
