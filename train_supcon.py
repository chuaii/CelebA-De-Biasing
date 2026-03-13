import argparse
from collections import Counter
import torch
import config as cfg
from dataset import get_loader
from model import FairClassifier
from loss import TotalLoss
from eval import evaluate
from utils import set_seed, get_device, log_epoch, BestTracker


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--group-balance",
        choices=["none", "oversampling", "reweighting"],
        default="oversampling",
        help="Group-balance method for training data/loss",
    )
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS, help=f"Number of epochs (default: {cfg.NUM_EPOCHS})")
    p.add_argument("--lambda-con", type=float, default=cfg.LAMBDA_CON, help=f"Contrastive loss weight (default: {cfg.LAMBDA_CON})")
    p.add_argument("--temperature", type=float, default=cfg.TEMPERATURE, help=f"SupCon temperature tau (default: {cfg.TEMPERATURE})")
    args = p.parse_args()

    group_balance_mode = args.group_balance

    set_seed()
    device = get_device()
    tag = f"FSC_{group_balance_mode}_{cfg.TARGET_ATTR}_vs_{cfg.SENSITIVE_ATTR}"
    lambda_con = args.lambda_con
    temperature = args.temperature
    epochs = args.epochs
    print(
        f"Fairness Sup_Con:  epochs={epochs}  lambda={lambda_con}  tau={temperature}  "
        f"group_balance={group_balance_mode}  warmup={cfg.WARMUP_EPOCHS}  device={device}"
    )

    loader_balance_mode = "oversampling" if group_balance_mode == "oversampling" else "none"
    train_loader = get_loader("train", cfg.BATCH_SIZE, group_balance_mode=loader_balance_mode)
    val_loader = get_loader("val", cfg.BATCH_SIZE)

    group_weights = None
    if group_balance_mode == "reweighting":
        group_counts = Counter(train_loader.dataset.groups)
        raw_weights = torch.tensor([1.0 / group_counts[g] for g in range(4)], dtype=torch.float32)
        group_weights = raw_weights / raw_weights.mean()
        print(f"Reweighting normalized group weights: {group_weights.tolist()}")

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=lambda_con, temperature=temperature, group_weights=group_weights).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": cfg.LR},
    ], weight_decay=cfg.WD)

    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[cfg.WARMUP_EPOCHS])

    tracker = BestTracker(tag, warmup_epochs=cfg.WARMUP_EPOCHS)
    for ep in range(epochs):
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
        log_epoch(ep, epochs, loss, m)
        tracker.update(model, m, ep)

    print(f"done. {tracker.summary()}")


if __name__ == "__main__":
    main()
