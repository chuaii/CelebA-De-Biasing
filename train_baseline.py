import argparse
import torch
import config as cfg
from dataset import get_loader
from model import FairClassifier
from loss import TotalLoss
from eval import evaluate
from utils import set_seed, get_device, log_epoch, BestTracker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()

    set_seed()
    device = get_device()
    epochs = args.epochs
    print(f"Baseline (ERM):  epochs={epochs}  warmup={cfg.WARMUP_EPOCHS}  device={device}")

    train_loader = get_loader("train", args.bs, group_balance_mode="none")
    val_loader = get_loader("val", args.bs)

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=0.0)  # only ERM，no FairSupCon

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=cfg.WD)

    # Warmup + CosineAnnealingWarmRestarts
    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[cfg.WARMUP_EPOCHS])

    tracker = BestTracker("baseline_"+cfg.TARGET_ATTR+"_vs_"+cfg.SENSITIVE_ATTR, warmup_epochs=cfg.WARMUP_EPOCHS)
    for ep in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets, _, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            emb, logits = model(images)
            loss, _, _ = criterion(logits, emb, targets, sensitives=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        loss = total_loss / n
        scheduler.step()
        m = evaluate(model, val_loader, device)
        log_epoch(ep, epochs, loss, m)
        tracker.update(model, m, ep)

    print(f"done. {tracker.summary()}")


if __name__ == "__main__":
    main()
