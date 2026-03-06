"""
Group DRO  (Sagawa et al., ICLR 2020)
Distributionally Robust Optimization over worst-case groups.

Each step:
  1. Compute per-group CE losses  L_g
  2. Update group weights:  q_g ← q_g · exp(η · L_g),  then normalize
  3. Backprop the weighted loss:  L = Σ_g  q_g · L_g

This drives the model to minimise the *worst* group loss.
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

import config
from dataset import get_loader
from model import FairClassifier
from eval import evaluate
from utils.utils import set_seed, get_device, log_epoch, save_best


def train_one_epoch(model, loader, optimizer, device, q, eta):
    model.train()
    ce = nn.CrossEntropyLoss(reduction="none")
    total_loss, n = 0.0, 0

    for images, targets, _, groups in loader:
        images, targets, groups = (
            images.to(device), targets.to(device), groups.to(device),
        )
        _, logits = model(images)
        per_sample = ce(logits, targets)

        group_losses = torch.zeros(4, device=device)
        for g in range(4):
            mask = groups == g
            if mask.any():
                group_losses[g] = per_sample[mask].mean()

        q = q * torch.exp(eta * group_losses.detach())
        q = q / q.sum()

        loss = (q * group_losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n, q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    p.add_argument("--dro_eta", type=float, default=0.01,
                   help="step size for group weight update")
    args = p.parse_args()

    set_seed()
    device = get_device()
    print(f"Group DRO  eta={args.dro_eta}  device={device}")

    train_loader = get_loader("train", args.bs)
    val_loader = get_loader("val", args.bs)

    model = FairClassifier().to(device)

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=config.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    q = torch.ones(4, device=device) / 4

    best_wga = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        loss, q = train_one_epoch(
            model, train_loader, optimizer, device, q, args.dro_eta,
        )
        scheduler.step()
        m = evaluate(model, val_loader, device)
        qstr = " ".join(f"{v:.3f}" for v in q.tolist())
        log_epoch(ep, args.epochs, time.time() - t0, loss, m, extra=f"q=[{qstr}]")
        best_wga = save_best(model, m["worst_group_acc"], best_wga, "groupdro")

    print(f"done. best wga={best_wga:.2%}")


if __name__ == "__main__":
    main()
