"""
评估脚本：整体准确率、分组准确率、公平性指标（DPD / EOD / EqOdd）。

公平性指标说明：
  - Demographic Parity Difference  (DPD)
  - Equal Opportunity Difference    (EOD)
  - Equalized Odds Difference       (max of |TPR gap|, |FPR gap|)
  - Per-sensitive-group accuracy / TPR / FPR
"""

import argparse
from collections import defaultdict
import torch
import config as cfg
from dataset import get_loader
from model import FairClassifier

# ---------- 公平性指标 ----------

def compute_fairness(preds, targets, sensitives):
    """
    对二值敏感属性计算公平性指标。
    返回包含 per-group 统计与差距指标的 dict。
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


def print_fairness_report(metrics):
    """打印详细公平性报告。"""
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


@torch.no_grad()
def collect_predictions(model, loader, device):
    """返回 (preds, targets, sensitives) 用于公平性计算。"""
    all_preds, all_targets, all_sensitives = [], [], []
    model.eval()
    for images, targets, sensitives, _ in loader:
        images = images.to(device)
        preds = model(images)[1].argmax(1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)
        all_sensitives.append(sensitives)
    return torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_sensitives)


# ---------- 主评估逻辑 ----------

@torch.no_grad()
def evaluate(model, loader, device):
    """返回 group accuracy 指标 + 公平性指标（DPD / EOD / EqOdd）。"""
    model.eval()
    correct, count = defaultdict(int), defaultdict(int)
    total_correct, total_count = 0, 0
    all_preds, all_targets, all_sensitives = [], [], []

    for images, targets, sensitives, groups in loader:
        images, targets, groups = images.to(device), targets.to(device), groups.to(device)
        preds = model(images)[1].argmax(1)
        hit = (preds == targets)

        total_correct += hit.sum().item()
        total_count += len(targets)
        for g in range(4):
            mask = groups == g
            correct[g] += (hit & mask).sum().item()
            count[g] += mask.sum().item()

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_sensitives.append(sensitives)

    group_acc = {g: correct[g] / max(count[g], 1) for g in range(4)}
    wg = min(group_acc, key=group_acc.get)

    fair = compute_fairness(
        torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_sensitives),
    )

    return {
        "overall_acc": total_correct / max(total_count, 1),
        "group_acc": group_acc,
        "worst_group_acc": group_acc[wg],
        "worst_group_id": wg,
        "dpd": fair["demographic_parity_diff"],
        "eod": fair["equal_opportunity_diff"],
        "eqodd": fair["equalized_odds_diff"],
        "fairness_metrics": fair,
    }


def main():
    p = argparse.ArgumentParser(description="评估模型：准确率与公平性指标")
    p.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--report", action="store_true", help="打印详细公平性报告")
    args = p.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = FairClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    m = evaluate(model, get_loader(args.split, args.bs), device)

    print(f"split={args.split}  overall={m['overall_acc']:.2%}  wga={m['worst_group_acc']:.2%}"
          f"  eod={m['eod']:.4f}  eqodd={m['eqodd']:.4f}  dpd={m['dpd']:.4f}")
    for g in range(4):
        tag = " <- worst" if g == m["worst_group_id"] else ""
        print(f"  {cfg.GROUP_NAMES[g]}: {m['group_acc'][g]:.2%}{tag}")

    if args.report:
        print_fairness_report(m["fairness_metrics"])


if __name__ == "__main__":
    main()
