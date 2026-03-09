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
    p.add_argument("--resampling", action="store_true", help="Use group-balanced resampling")
    p.add_argument("--no-resampling", action="store_true", help="Use uniform sampling (FairSupCon-only)")
    args = p.parse_args()

    if args.resampling and args.no_resampling:
        raise ValueError("Cannot specify both --resampling and --no-resampling")

    # Debias 默认；resample 开关：有 --resampling 则平衡采样，有 --no-resampling 则均匀采样，都没有则默认平衡采样
    balanced = args.resampling if args.resampling else (not args.no_resampling)

    set_seed()
    device = get_device()
    tag = "resample" if balanced else "no_resample"
    print(f"Debias:  lambda={cfg.LAMBDA_CON}  tau={cfg.TEMPERATURE}  balanced={balanced}  warmup={cfg.WARMUP_EPOCHS}  device={device}")

    train_loader = get_loader("train", cfg.BATCH_SIZE, balanced=balanced)
    val_loader = get_loader("val", cfg.BATCH_SIZE)

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=cfg.LAMBDA_CON, temperature=cfg.TEMPERATURE)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": cfg.LR},
    ], weight_decay=cfg.WD)

    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[cfg.WARMUP_EPOCHS])

    tracker = BestTracker(tag)
    for ep in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets, sensitives, _ in train_loader:
            imgs, tgts, sens = images.to(device), targets.to(device), sensitives.to(device)
            emb, logits = model(imgs)
            loss, _, _ = criterion(logits, emb, tgts, sens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        loss = total_loss / n
        scheduler.step()
        m = evaluate(model, val_loader, device)
        log_epoch(ep, cfg.NUM_EPOCHS, loss, m)
        tracker.update(model, m)

    print(f"done. {tracker.summary()}")


if __name__ == "__main__":
    main()
