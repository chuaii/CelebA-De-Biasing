import argparse
import csv
import os
from collections import Counter
import torch
import config as cfg
from dataset import get_loader
from fair_supcon import FairClassifier, TotalLoss
from eval import evaluate
from utils import set_seed, get_device, log_epoch, BestTracker

_GN = cfg.GROUP_NAMES  # {0: "...", 1: "...", 2: "...", 3: "..."}
# Short stems for default CSV names under training/ (see training/*.csv)
_TRAINING_CSV_STEM = {
    ("Blond_Hair", "Male"): "blond_male",
    ("Mouth_Slightly_Open", "Smiling"): "mouth_smiling",
    ("Smiling", "High_Cheekbones"): "smiling_highcheekbones",
}

CSV_FIELDS = [
    "method", "λ", "group_balance", "epoch",
    "train_loss", "overall_acc", "wga", "worst_group", "eqodd",
    *[f"acc_{_GN[g]}" for g in range(4)],
]


def default_training_csv_path() -> str:
    key = (cfg.TARGET_ATTR, cfg.SENSITIVE_ATTR)
    stem = _TRAINING_CSV_STEM.get(key)
    if stem is None:
        t = cfg.TARGET_ATTR.replace("_", "").lower()
        s = cfg.SENSITIVE_ATTR.replace("_", "").lower()
        stem = f"{t}_{s}"
    return os.path.join(cfg.ROOT, "training", f"training_{stem}.csv")


def append_csv(path, row):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lambda-con", type=float, default=0.0, help="对比损失权重，0 = baseline (ERM only)")
    p.add_argument("--temperature", type=float, default=cfg.TEMPERATURE)
    p.add_argument("--group-balance", choices=["none", "oversampling", "reweighting"], default="none")
    p.add_argument("--csv", type=str, default=None, help="CSV path to append per-epoch metrics")
    args = p.parse_args()

    if args.csv is None:
        os.makedirs(os.path.join(cfg.ROOT, "training"), exist_ok=True)
        args.csv = default_training_csv_path()

    set_seed()
    device = get_device()
    lambda_con = args.lambda_con
    group_balance = args.group_balance

    is_baseline = lambda_con == 0.0
    method = "ERM" if is_baseline else f"FSC"
    tag = (f"baseline_{cfg.TARGET_ATTR}_vs_{cfg.SENSITIVE_ATTR}" if is_baseline
           else f"FSC_{group_balance}_{cfg.TARGET_ATTR}_{cfg.SENSITIVE_ATTR}")

    print(f"{'Baseline' if is_baseline else 'FairSupCon'}:  "
          f"epochs={args.epochs}  lambda={lambda_con}  tau={args.temperature}  "
          f"group_balance={group_balance}  warmup={cfg.WARMUP_EPOCHS}  device={device}")

    loader_balance = "oversampling" if group_balance == "oversampling" else "none"
    train_loader = get_loader("train", args.bs, group_balance_mode=loader_balance)
    val_loader = get_loader("val", args.bs)

    group_weights = None
    if group_balance == "reweighting":
        group_counts = Counter(train_loader.dataset.groups)
        raw_weights = torch.tensor([1.0 / group_counts[g] for g in range(4)], dtype=torch.float32)
        group_weights = raw_weights / raw_weights.mean()
        print(f"Reweighting normalized group weights: {group_weights.tolist()}")

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=lambda_con, temperature=args.temperature, group_weights=group_weights).to(device) # Total Loss Function

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=cfg.WD)

    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[cfg.WARMUP_EPOCHS])

    tracker = BestTracker(tag, warmup_epochs=cfg.WARMUP_EPOCHS)
    for ep in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets, sensitives, _ in train_loader:
            imgs, tgts, sens = images.to(device), targets.to(device), sensitives.to(device)
            optimizer.zero_grad()
            emb, logits = model(imgs)
            loss, _, _ = criterion(logits, emb, tgts, sens)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        avg_loss = total_loss / n
        m = evaluate(model, val_loader, device)
        log_epoch(ep, args.epochs, avg_loss, m)
        tracker.update(model, m, ep)

        if args.csv:
            append_csv(args.csv, {
                "method": method,
                "λ": lambda_con,
                "group_balance": group_balance,
                "epoch": ep + 1,
                "train_loss": f"{avg_loss:.4f}",
                "overall_acc": f"{m['overall_acc']:.4f}",
                "wga": f"{m['worst_group_acc']:.4f}",
                "worst_group": _GN[m["worst_group_id"]],
                "eqodd": f"{m['eqodd']:.4f}",
                **{f"acc_{_GN[g]}": f"{m['group_acc'][g]:.4f}" for g in range(4)},
            })

    print(f"done. {tracker.summary()}")


if __name__ == "__main__":
    main()
