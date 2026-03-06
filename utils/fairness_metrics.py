"""
Fairness evaluation script — standalone, does NOT modify any existing file.

Metrics computed:
  - Demographic Parity Difference  (DPD)
  - Equal Opportunity Difference    (EOD)
  - Equalized Odds Difference       (max of |TPR gap|, |FPR gap|)
  - Per-sensitive-group accuracy / TPR / FPR
"""

import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import config
from dataset import get_loader
from model import FairClassifier


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return parallel lists of (prediction, target, sensitive)."""
    all_preds, all_targets, all_sensitives = [], [], []
    model.eval()
    for images, targets, sensitives, _ in loader:
        images = images.to(device)
        preds = model(images)[1].argmax(1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)
        all_sensitives.append(sensitives)
    return torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_sensitives)


def compute_fairness(preds, targets, sensitives):
    """
    Compute fairness metrics for a binary sensitive attribute.

    Returns a dict with per-group stats and gap metrics.
    """
    groups = sorted(sensitives.unique().tolist())
    assert len(groups) == 2, f"Expected binary sensitive attr, got {groups}"
    g0, g1 = groups  # 0 = Female, 1 = Male

    stats = {}
    for g in groups:
        mask = sensitives == g
        g_preds = preds[mask]
        g_targets = targets[mask]

        n = mask.sum().item()
        acc = (g_preds == g_targets).float().mean().item()
        pos_rate = (g_preds == 1).float().mean().item()

        tp = ((g_preds == 1) & (g_targets == 1)).sum().item()
        fp = ((g_preds == 1) & (g_targets == 0)).sum().item()
        fn = ((g_preds == 0) & (g_targets == 1)).sum().item()
        tn = ((g_preds == 0) & (g_targets == 0)).sum().item()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        stats[g] = dict(n=n, acc=acc, pos_rate=pos_rate, tpr=tpr, fpr=fpr,
                        tp=tp, fp=fp, fn=fn, tn=tn)

    dpd = abs(stats[g0]["pos_rate"] - stats[g1]["pos_rate"])
    eod = abs(stats[g0]["tpr"] - stats[g1]["tpr"])
    eqodd = max(eod, abs(stats[g0]["fpr"] - stats[g1]["fpr"]))

    return {
        "group_stats": stats,
        "demographic_parity_diff": dpd,
        "equal_opportunity_diff": eod,
        "equalized_odds_diff": eqodd,
    }


def print_report(metrics):
    gs = metrics["group_stats"]
    sensitive_names = {0: "Female", 1: "Male"}

    print("=" * 62)
    print("           Fairness Evaluation Report")
    print("=" * 62)

    header = f"{'Group':>10} {'N':>7} {'Acc':>8} {'P(Y=1)':>8} {'TPR':>8} {'FPR':>8}"
    print(header)
    print("-" * 62)
    for g in sorted(gs):
        s = gs[g]
        print(f"{sensitive_names[g]:>10} {s['n']:>7d} {s['acc']:>8.2%} "
              f"{s['pos_rate']:>8.4f} {s['tpr']:>8.2%} {s['fpr']:>8.2%}")

    print("-" * 62)
    print(f"  Demographic Parity Diff (DPD) : {metrics['demographic_parity_diff']:.4f}")
    print(f"  Equal Opportunity Diff  (EOD) : {metrics['equal_opportunity_diff']:.4f}")
    print(f"  Equalized Odds Diff           : {metrics['equalized_odds_diff']:.4f}")
    print("=" * 62)
    print()
    print("  DPD = |P(Ŷ=1|Female) − P(Ŷ=1|Male)|          → 0 is fair")
    print("  EOD = |TPR_Female − TPR_Male|                  → 0 is fair")
    print("  EqOdd = max(|ΔTPR|, |ΔFPR|)                   → 0 is fair")
    print()


def main():
    p = argparse.ArgumentParser(description="Fairness metrics evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    args = p.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = FairClassifier().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )

    loader = get_loader(args.split, args.bs)
    preds, targets, sensitives = collect_predictions(model, loader, device)
    metrics = compute_fairness(preds, targets, sensitives)
    print_report(metrics)

    return metrics


if __name__ == "__main__":
    main()
