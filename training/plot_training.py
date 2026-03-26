from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Plot training curves for bias & de-biasing comparison.
Usage:
  cd D:\Code\DeepLearning\CelebA\training
  python plot_training.py                       # all tasks
  python plot_training.py --task blond_male     # single task
  python plot_training.py --task mouth_smiling
"""

SCRIPT_DIR = Path(__file__).parent

METHODS = ["Baseline (ERM)", "FSC (Unbalanced)", "FSC (Oversampling)", "FSC (Reweighting)"]
COLORS = {
    "Baseline (ERM)":    "#1f77b4",
    "FSC (Unbalanced)":  "#ff7f0e",
    "FSC (Oversampling)":"#2ca02c",
    "FSC (Reweighting)": "#d62728",
}

TASK_CONFIGS = {
    "blond_male": {
        "csv": "training_blond_male.csv",
        "out": "training_log_val_blond_male.png",
        "group_cols": [
            "acc_NonBlond_Female", "acc_NonBlond_Male",
            "acc_Blond_Female", "acc_Blond_Male",
        ],
        "group_labels": ["NB / F", "NB / M", "B / F", "B / M"],
        "suptitle": "Blond Hair \u00d7 Male \u2014 Bias & De-biasing Comparison",
        "bias_title": "(a) Bias in Baseline\n(B = Blond, NB = Non-Blond, F = Female, M = Male)",
        "bias_ylim": (20, 108),
        "final_ylim": (20, 105),
    },
    "mouth_smiling": {
        "csv": "training_mouth_smiling.csv",
        "out": "training_log_val_mouth_smiling.png",
        "group_cols": [
            "acc_MouthNonOpen_NonSmiling", "acc_MouthNonOpen_Smiling",
            "acc_MouthOpen_NonSmiling", "acc_MouthOpen_Smiling",
        ],
        "group_labels": ["MO- / S-", "MO- / S+", "MO+ / S-", "MO+ / S+"],
        "suptitle": "Mouth Slightly Open \u00d7 Smiling \u2014 Bias & De-biasing Comparison",
        "bias_title": "(a) Bias in Baseline\n(MO+ = Mouth Open, S+ = Smiling)",
        "bias_ylim": (75, 102),
        "final_ylim": (78, 100),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves for bias & de-biasing comparison")
    p.add_argument("--task", choices=sorted(TASK_CONFIGS), default=None,
                   help="Task to plot (default: all)")
    return p.parse_args()


def method_xtick_label(label: str) -> str:
    s = str(label).strip()
    if not s.startswith("FSC"):
        return s
    return s + "\n" + r"$\mathbf{Ours}$"


def style_method_xticklabels(ax: plt.Axes) -> None:
    for lbl in ax.get_xticklabels():
        lbl.set_multialignment("center")
        lbl.set_ha("center")


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    lam_col = [c for c in df.columns if "lambda" in c.lower() or c in ("\ufeff\u03bb", "\u03bb", "?", "??")]
    if lam_col:
        df = df.rename(columns={lam_col[0]: "lambda"})

    def label(row):
        if row["method"] == "ERM":
            return "Baseline (ERM)"
        gb = str(row.get("group_balance", "none")).strip().lower()
        if gb in ("none", "nan", ""):
            return "FSC (Unbalanced)"
        if gb == "oversampling":
            return "FSC (Oversampling)"
        if gb == "reweighting":
            return "FSC (Reweighting)"
        return f"FSC ({gb})"

    df["label"] = df.apply(label, axis=1)
    return df


def final_row(df: pd.DataFrame, label: str) -> pd.Series:
    sub = df[df["label"] == label]
    return sub.loc[sub["epoch"].idxmax()]


def plot_bias(ax: plt.Axes, df: pd.DataFrame, cfg: dict) -> None:
    group_cols = cfg["group_cols"]
    group_labels = cfg["group_labels"]

    row = final_row(df, "Baseline (ERM)")
    accs = [row[c] * 100.0 for c in group_cols]

    worst_idx = int(np.argmin(accs))
    best_idx = int(np.argmax(accs))
    bar_colors = ["#aec6e8"] * len(group_cols)
    bar_colors[worst_idx] = "#d62728"
    bar_colors[best_idx] = "#2ca02c"
    labels = list(group_labels)
    labels[worst_idx] += "\n(worst)"

    x = np.arange(len(group_labels))
    bars = ax.bar(x, accs, color=bar_colors, edgecolor="white", width=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    y_lo = accs[worst_idx]
    y_hi = accs[best_idx]
    ax.annotate("", xy=(best_idx, y_hi - 1), xytext=(worst_idx, y_lo + 1),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    mid_y = (y_lo + y_hi) / 2
    ax.text((best_idx + worst_idx) / 2, mid_y,
            f"gap\n{y_hi - y_lo:.1f}pp", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*cfg["bias_ylim"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(cfg["bias_title"], fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    legend_patches = [
        mpatches.Patch(color="#d62728", label="Worst group (biased)"),
        mpatches.Patch(color="#2ca02c", label="Best group (spurious)"),
        mpatches.Patch(color="#aec6e8", label="Other groups"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, frameon=False)


def plot_wga_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    for method in METHODS:
        sub = df[df["label"] == method].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(sub["epoch"], sub["wga"] * 100.0,
                marker="o", markersize=4, label=method,
                color=COLORS[method], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("WGA (%)")
    ax.set_title("(b) Worst-Group Accuracy over Training\n(higher is better)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xticks(range(1, 11))


def plot_final_bars(ax: plt.Axes, df: pd.DataFrame, cfg: dict) -> None:
    overall = [final_row(df, m)["overall_acc"] * 100.0 for m in METHODS]
    wga     = [final_row(df, m)["wga"] * 100.0         for m in METHODS]

    x = np.arange(len(METHODS))
    w = 0.35
    b1 = ax.bar(x - w / 2, overall, w, label="Overall Acc", color="#4C72B0")
    b2 = ax.bar(x + w / 2, wga,     w, label="WGA",         color="#55A868")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([method_xtick_label(m) for m in METHODS], rotation=12, fontsize=9)
    style_method_xticklabels(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(*cfg["final_ylim"])
    ax.set_title("(c) Final Accuracy Comparison\n(epoch 10)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)


def plot_eqodd_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    for method in METHODS:
        sub = df[df["label"] == method].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(sub["epoch"], sub["eqodd"] * 100.0,
                marker="o", markersize=4, label=method,
                color=COLORS[method], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Equalized Odds Gap (%)")
    ax.set_title("(d) Fairness: Equalized Odds Gap\n(lower is better)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xticks(range(1, 11))


def plot_task(task_name: str) -> Path:
    cfg = TASK_CONFIGS[task_name]
    csv_path = SCRIPT_DIR / cfg["csv"]
    out_path = SCRIPT_DIR / cfg["out"]

    df = load(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(cfg["suptitle"], fontsize=14, fontweight="bold", y=0.99)

    plot_bias(axes[0, 0], df, cfg)
    plot_wga_curves(axes[0, 1], df)
    plot_final_bars(axes[1, 0], df, cfg)
    plot_eqodd_curves(axes[1, 1], df)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    tasks = [args.task] if args.task else list(TASK_CONFIGS)
    for task in tasks:
        out = plot_task(task)
        print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
