import argparse
import os
import random
import time

import numpy as np
import torch

import config
from dataset import get_loader
from model import FairClassifier
from loss import TotalLoss
from eval import evaluate


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_one_epoch(model, loader, criterion, optimizer, device, use_sensitive=False):
    model.train()
    total_loss, n = 0.0, 0
    for images, targets, sensitives, _ in loader:
        images, targets = images.to(device), targets.to(device)
        sens = sensitives.to(device) if use_sensitive else None
        emb, logits = model(images)
        loss, _, _ = criterion(logits, emb, targets, sens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "debias"], default="debias")
    p.add_argument("--lambda_con", type=float, default=None)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    args = p.parse_args()

    is_debias = args.mode == "debias"
    lam = args.lambda_con if args.lambda_con is not None else (config.LAMBDA_CON if is_debias else 0.0)
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"mode={args.mode}  lambda={lam}  balanced={is_debias}  device={device}")

    train_loader = get_loader("train", args.bs, balanced=is_debias)
    val_loader = get_loader("val", args.bs)

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=lam)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": config.LR_BACKBONE},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=config.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    best_wga, no_improve = 0.0, 0

    for ep in range(0, args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_sensitive=is_debias)
        scheduler.step()

        m = evaluate(model, val_loader, device)
        wga = m["worst_group_acc"]

        print(f"[{ep+1}/{args.epochs}] {time.time()-t0:.0f}s  loss={loss:.4f}  acc={m['overall_acc']:.2%}  wga={wga:.2%}")
        for g in range(4):
            print(f"  {config.GROUP_NAMES[g]}: {m['group_acc'][g]:.2%}")

        if wga > best_wga:
            best_wga = wga
            no_improve = 0
            path = os.path.join(config.CKPT_DIR, f"best_{args.mode}.pt")
            torch.save(model.state_dict(), path)
            print(f"  -> saved {path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stopping at epoch {ep+1}, no improvement for {args.patience} epochs")
                break

    print(f"done. best wga={best_wga:.2%}")


if __name__ == "__main__":
    main()
