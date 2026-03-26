from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import config as cfg
from dataset import get_loader
from eval import collect_predictions, compute_metrics_from_predictions
from fair_supcon import FairClassifier
from utils import get_device, set_seed

"""
Evaluate the fairness of the checkpoints.
  You can use the following command to evaluate the fairness of the checkpoints:
  cd D:\Code\DeepLearning\CelebA
  python bootstrap_eval.py
"""

DEFAULT_BOOTSTRAP_SEEDS = [5, 17, 29, 41, 53]
GROUP_BALANCE_CHOICES = ("none", "oversampling", "reweighting")
METRIC_NAMES = ["overall_acc", "wga", "eqodd", "acc_g0", "acc_g1", "acc_g2", "acc_g3"]

TASK_SPECS = [
    {
        "task": "BlondHair_Male",
        "target_attr": "Blond_Hair",
        "sensitive_attr": "Male",
        "group_names": {
            0: "NonBlond_Female",
            1: "NonBlond_Male",
            2: "Blond_Female",
            3: "Blond_Male",
        },
    },
    {
        "task": "MouthOpen_Smiling",
        "target_attr": "Mouth_Slightly_Open",
        "sensitive_attr": "Smiling",
        "group_names": {
            0: "MouthClosed_NonSmiling",
            1: "MouthClosed_Smiling",
            2: "MouthOpen_NonSmiling",
            3: "MouthOpen_Smiling",
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap CI evaluation for WGA checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/wga"))
    parser.add_argument("--checkpoint-glob", default="*.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--bootstrap-seeds", nargs="+", type=int, default=DEFAULT_BOOTSTRAP_SEEDS)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/bootstrap_ci_summary.csv"))
    parser.add_argument("--save-raw", type=Path, default=None)
    return parser.parse_args()


def set_task_config(task_spec):
    cfg.TARGET_ATTR = task_spec["target_attr"]
    cfg.SENSITIVE_ATTR = task_spec["sensitive_attr"]
    cfg.GROUP_NAMES = task_spec["group_names"]


def infer_task_spec(checkpoint_name):
    for spec in TASK_SPECS:
        baseline_tag = f"baseline_{spec['target_attr']}_vs_{spec['sensitive_attr']}"
        fsc_tag = f"{spec['target_attr']}_{spec['sensitive_attr']}"
        if baseline_tag in checkpoint_name or checkpoint_name.endswith(f"{fsc_tag}_wga.pt"):
            return spec
    raise ValueError(f"Cannot infer task from checkpoint name: {checkpoint_name}")


def parse_checkpoint_metadata(checkpoint_path, task_spec):
    stem = checkpoint_path.stem
    if stem.startswith("best_baseline_"):
        return {
            "task": task_spec["task"],
            "target_attr": task_spec["target_attr"],
            "sensitive_attr": task_spec["sensitive_attr"],
            "method": "ERM",
            "group_balance": "none",
            "method_label": "Baseline (ERM)",
        }

    if not stem.startswith("best_FSC_"):
        raise ValueError(f"Unsupported checkpoint naming: {checkpoint_path.name}")

    suffix = stem[len("best_FSC_") :]
    group_balance = next((name for name in GROUP_BALANCE_CHOICES if suffix.startswith(f"{name}_")), None)
    if group_balance is None:
        raise ValueError(f"Cannot infer group balance from checkpoint name: {checkpoint_path.name}")

    label_map = {
        "none": "FSC (Unbalanced)",
        "oversampling": "FSC (Oversampling)",
        "reweighting": "FSC (Reweighting)",
    }
    return {
        "task": task_spec["task"],
        "target_attr": task_spec["target_attr"],
        "sensitive_attr": task_spec["sensitive_attr"],
        "method": "FSC",
        "group_balance": group_balance,
        "method_label": label_map[group_balance],
    }


def collect_point_metrics(metrics):
    row = {
        "overall_acc": metrics["overall_acc"],
        "wga": metrics["worst_group_acc"],
        "eqodd": metrics["eqodd"],
        "worst_group_id": metrics["worst_group_id"],
        "worst_group_name": cfg.GROUP_NAMES[metrics["worst_group_id"]],
    }
    row.update({f"acc_g{g}": metrics["group_acc"][g] for g in range(4)})
    return row


def bootstrap_metrics(preds, targets, sensitives, groups, bootstrap_seeds, num_bootstrap):
    sample_count = len(preds)
    raw_rows = []

    for bootstrap_seed in bootstrap_seeds:
        rng = np.random.default_rng(bootstrap_seed)
        for bootstrap_iter in range(num_bootstrap):
            indices = torch.from_numpy(rng.choice(sample_count, size=sample_count, replace=True)).long()
            metrics = compute_metrics_from_predictions(
                preds[indices],
                targets[indices],
                sensitives[indices],
                groups[indices],
            )
            metric_row = collect_point_metrics(metrics)
            metric_row["bootstrap_seed"] = bootstrap_seed
            metric_row["bootstrap_iter"] = bootstrap_iter + 1
            raw_rows.append(metric_row)

    return raw_rows


def summarize_bootstrap(raw_rows):
    summary = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray([row[metric_name] for row in raw_rows], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_ci_low"] = float(np.quantile(values, 0.025))
        summary[f"{metric_name}_ci_high"] = float(np.quantile(values, 0.975))
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(checkpoint_path, split, batch_size, loader_cache, device):
    task_spec = infer_task_spec(checkpoint_path.name)
    set_task_config(task_spec)

    loader_key = (task_spec["task"], split, batch_size)
    if loader_key not in loader_cache:
        loader_cache[loader_key] = get_loader(split, batch_size)

    model = FairClassifier().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    preds, targets, sensitives = collect_predictions(model, loader_cache[loader_key], device)
    groups = targets * 2 + sensitives
    return task_spec, preds, targets, sensitives, groups


def main():
    args = parse_args()
    set_seed()
    device = get_device()

    checkpoints = sorted(args.checkpoint_dir.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {args.checkpoint_dir} with pattern {args.checkpoint_glob}"
        )

    summary_rows = []
    raw_rows = []
    loader_cache = {}
    total_bootstrap = len(args.bootstrap_seeds) * args.num_bootstrap

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        task_spec, preds, targets, sensitives, groups = load_predictions(
            checkpoint_path, args.split, args.bs, loader_cache, device
        )
        metadata = parse_checkpoint_metadata(checkpoint_path, task_spec)
        point_metrics = collect_point_metrics(
            compute_metrics_from_predictions(preds, targets, sensitives, groups)
        )
        checkpoint_raw_rows = bootstrap_metrics(
            preds, targets, sensitives, groups, args.bootstrap_seeds, args.num_bootstrap
        )

        for row in checkpoint_raw_rows:
            raw_rows.append(
                {
                    "checkpoint": checkpoint_path.name,
                    **metadata,
                    "split": args.split,
                    **{f"group{g}_name": task_spec["group_names"][g] for g in range(4)},
                    **row,
                }
            )

        summary_rows.append(
            {
                "checkpoint": checkpoint_path.name,
                **metadata,
                "split": args.split,
                "n_samples": len(preds),
                "bootstrap_total": total_bootstrap,
                "bootstrap_seeds": ";".join(str(seed) for seed in args.bootstrap_seeds),
                **{f"group{g}_name": task_spec["group_names"][g] for g in range(4)},
                **{f"point_{key}": value for key, value in point_metrics.items()},
                **summarize_bootstrap(checkpoint_raw_rows),
            }
        )

        print(
            f"[{idx}/{len(checkpoints)}] {checkpoint_path.name}  "
            f"point_wga={point_metrics['wga']:.4f}  point_eqodd={point_metrics['eqodd']:.4f}  "
            f"bootstrap={total_bootstrap}"
        )

    write_csv(args.out_csv, summary_rows)
    print(f"Saved summary CSV: {args.out_csv}")

    if args.save_raw is not None:
        write_csv(args.save_raw, raw_rows)
        print(f"Saved raw CSV: {args.save_raw}")


if __name__ == "__main__":
    main()
