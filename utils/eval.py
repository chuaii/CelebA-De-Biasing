import argparse
from collections import defaultdict

import torch

import config
from dataset import get_loader
from model import FairClassifier


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, count = defaultdict(int), defaultdict(int)
    total_correct, total_count = 0, 0

    for images, targets, _, groups in loader:
        images, targets, groups = images.to(device), targets.to(device), groups.to(device)
        preds = model(images)[1].argmax(1)
        hit = (preds == targets)

        total_correct += hit.sum().item()
        total_count += len(targets)
        for g in range(4):
            mask = groups == g
            correct[g] += (hit & mask).sum().item()
            count[g] += mask.sum().item()

    group_acc = {g: correct[g] / max(count[g], 1) for g in range(4)}
    wg = min(group_acc, key=group_acc.get)
    return {
        "overall_acc": total_correct / max(total_count, 1),
        "group_acc": group_acc,
        "worst_group_acc": group_acc[wg],
        "worst_group_id": wg,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    args = p.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = FairClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    m = evaluate(model, get_loader(args.split, args.bs), device)

    print(f"split={args.split}  overall={m['overall_acc']:.2%}  wga={m['worst_group_acc']:.2%}")
    for g in range(4):
        tag = " <- worst" if g == m["worst_group_id"] else ""
        print(f"  {config.GROUP_NAMES[g]}: {m['group_acc'][g]:.2%}{tag}")


if __name__ == "__main__":
    main()
