"""
LfF 模型评估：逐组准确率 + Worst-Group Accuracy + Unbiased Accuracy。
"""
import argparse
from collections import defaultdict

import torch

import LfF_config as cfg
from LfF_dataset import get_eval_loader
from LfF_model import LfFClassifier


@torch.no_grad()
def lff_evaluate(model, loader, device):
    """
    返回:
      overall_acc     — 全样本准确率
      group_acc       — dict {0..3: float}，四组各自准确率
      worst_group_acc — 四组中最低
      worst_group_id  — 最差组 ID
      unbiased_acc    — 四组准确率的算术平均（消除组别大小影响）
    """
    model.eval()
    correct, count = defaultdict(int), defaultdict(int)
    total_correct, total_count = 0, 0

    for batch in loader:
        images, targets, _, groups = batch[:4]
        images = images.to(device)
        targets = targets.to(device)
        groups = groups.to(device)

        preds = model(images).argmax(1)
        hit = preds == targets

        total_correct += hit.sum().item()
        total_count += len(targets)
        for g in range(4):
            mask = groups == g
            correct[g] += (hit & mask).sum().item()
            count[g] += mask.sum().item()

    group_acc = {g: correct[g] / max(count[g], 1) for g in range(4)}
    wg = min(group_acc, key=group_acc.get)
    unbiased = sum(group_acc.values()) / 4.0

    return {
        "overall_acc": total_correct / max(total_count, 1),
        "group_acc": group_acc,
        "worst_group_acc": group_acc[wg],
        "worst_group_id": wg,
        "unbiased_acc": unbiased,
    }


def print_metrics(title, m):
    """格式化输出评估指标。"""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  Overall Accuracy : {m['overall_acc']:.2%}")
    print(f"  Worst-Group Acc  : {m['worst_group_acc']:.2%}  "
          f"({cfg.GROUP_NAMES[m['worst_group_id']]})")
    print(f"  Unbiased Accuracy: {m['unbiased_acc']:.2%}")
    print(f"  ---")
    for g in range(4):
        tag = " <- worst" if g == m["worst_group_id"] else ""
        print(f"  {cfg.GROUP_NAMES[g]:20s}: {m['group_acc'][g]:.2%}{tag}")
    print()


def main():
    p = argparse.ArgumentParser(description="LfF 模型评估")
    p.add_argument("--checkpoint", required=True, help="模型权重路径")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bs", type=int, default=cfg.MAIN_BATCH_SIZE)
    args = p.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = LfFClassifier().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )

    loader = get_eval_loader(args.split, args.bs)
    m = lff_evaluate(model, loader, device)
    print_metrics(f"LfF Model ({args.split})", m)


if __name__ == "__main__":
    main()
