import argparse
import time
import torch
import config
from dataset import get_loader
from model import FairClassifier
from loss import TotalLoss
from eval import evaluate
from utils.utils import set_seed, get_device, log_epoch, save_best


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
    args = p.parse_args()

    is_debias = args.mode == "debias"
    lam = args.lambda_con if args.lambda_con is not None else (config.LAMBDA_CON if is_debias else 0.0)
    set_seed()
    device = get_device()
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

    best_wga = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_sensitive=is_debias)
        scheduler.step()
        m = evaluate(model, val_loader, device)
        log_epoch(ep, args.epochs, time.time() - t0, loss, m)
        best_wga = save_best(model, m["worst_group_acc"], best_wga, args.mode)

    print(f"done. best wga={best_wga:.2%}")


if __name__ == "__main__":
    main()
