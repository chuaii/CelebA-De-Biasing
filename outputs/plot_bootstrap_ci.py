from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

"""
from bootstrap_ci_summary.csv to build the figure.
You can use the following command to build the figure:
  cd D:\Code\DeepLearning\CelebA\outputs
  python plot_bootstrap_ci.py
You can also use the following command to build the figure for a specific task:
  python plot_bootstrap_ci.py --task MouthOpen_Smiling
  python plot_bootstrap_ci.py --task BlondHair_Male
"""

CSV_PATH = Path(__file__).parent / "bootstrap_ci_summary.csv"
OUT_DIR = Path(__file__).parent

METHOD_ORDER = [
    "Baseline (ERM)",
    "FSC (Unbalanced)",
    "FSC (Oversampling)",
    "FSC (Reweighting)",
]

TASK_TITLES = {
    "BlondHair_Male": "Blond x Male - Accuracy Breakdown by Demographic Groups",
    "MouthOpen_Smiling": "Mouth x Smiling - Accuracy Breakdown by Demographic Groups",
}

TASK_OUTPUTS = {
    "BlondHair_Male": "bootstrap_blondhair_male_acc.png",
    "MouthOpen_Smiling": "bootstrap_mouthopen_smiling_acc.png",
}

BAR_COLORS = {
    "overall": "#34495e",
    "g0": "#5dade2",
    "g1": "#2bb9a8",
    "g2": "#e74c3c",
    "g3": "#f4d03f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grouped bootstrap CI accuracy figures")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--task", choices=sorted(TASK_TITLES), default=None)
    return parser.parse_args()


def method_xtick_label(label: str) -> str:
    label = str(label).strip()
    if not label.startswith("FSC"):
        return label
    return label + "\n" + r"$\mathbf{Ours}$"


def method_xtick_labels(labels) -> list[str]:
    return [method_xtick_label(label) for label in labels]


def style_method_xticklabels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels():
        label.set_multialignment("center")
        label.set_ha("center")


def _safe_task_name(task: str) -> str:
    return task.lower().replace(" ", "_")


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["method_label"] = pd.Categorical(df["method_label"], METHOD_ORDER, ordered=True)
    df = df.sort_values(["task", "method_label"]).reset_index(drop=True)
    return df


def _task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = df[df["task"] == task].copy()
    task_df["method_label"] = pd.Categorical(task_df["method_label"], METHOD_ORDER, ordered=True)
    task_df = task_df.sort_values("method_label").reset_index(drop=True)
    if task_df.empty:
        raise ValueError(f"No rows found for task: {task}")
    return task_df


def metric_mean_and_error(task_df: pd.DataFrame, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    mean = task_df[f"{metric_name}_mean"].to_numpy(dtype=float)
    low = task_df[f"{metric_name}_ci_low"].to_numpy(dtype=float)
    high = task_df[f"{metric_name}_ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([mean - low, high - mean])
    return mean, yerr


def add_gap_annotation(
    ax: plt.Axes,
    group_centers: np.ndarray,
    group_means: list[np.ndarray],
    group_errors: list[np.ndarray],
    method_idx: int,
) -> None:
    heights = np.array([values[method_idx] for values in group_means], dtype=float)
    lowers = np.array([errors[0, method_idx] for errors in group_errors], dtype=float)
    uppers = np.array([errors[1, method_idx] for errors in group_errors], dtype=float)

    max_idx = int(np.argmax(heights))
    min_idx = int(np.argmin(heights))
    if max_idx == min_idx:
        return

    h_max = heights[max_idx]
    h_min = heights[min_idx]
    x_max = group_centers[max_idx]
    x_min = group_centers[min_idx]

    ax.text(
        x_max,
        h_max + uppers[max_idx] + 0.012,
        f"{100 * h_max:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    if h_min >= 0.10:
        ax.text(
            x_min,
            max(h_min * 0.92, h_min - 0.028),
            f"{100 * h_min:.1f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            color="0.15",
        )
    else:
        ax.text(
            x_min,
            h_min + uppers[min_idx] + 0.012,
            f"{100 * h_min:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    y_max = h_max - lowers[max_idx]
    y_min = h_min + uppers[min_idx] + 0.007
    arrow = FancyArrowPatch(
        (x_max, y_max),
        (x_min, y_min),
        arrowstyle="<->",
        mutation_scale=9,
        linewidth=1.0,
        color="black",
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    )
    ax.add_patch(arrow)

    gap_label_y = max(y_max, y_min) + 0.08

    ax.text(
        (x_max + x_min) / 2,
        gap_label_y,
        f"gap {100 * (h_max - h_min):.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#f5e6d3",
            edgecolor="0.45",
            linewidth=0.6,
        ),
        zorder=7,
    )


def plot_task(df: pd.DataFrame, task: str) -> Path:
    task_df = _task_frame(df, task)
    group_names = [task_df.iloc[0][f"group{i}_name"] for i in range(4)]

    overall_mean, overall_err = metric_mean_and_error(task_df, "overall_acc")
    g0_mean, g0_err = metric_mean_and_error(task_df, "acc_g0")
    g1_mean, g1_err = metric_mean_and_error(task_df, "acc_g1")
    g2_mean, g2_err = metric_mean_and_error(task_df, "acc_g2")
    g3_mean, g3_err = metric_mean_and_error(task_df, "acc_g3")

    group_means = [g0_mean, g1_mean, g2_mean, g3_mean]
    group_errors = [g0_err, g1_err, g2_err, g3_err]

    x = np.arange(len(task_df))
    width = 0.16
    error_style = dict(ecolor="black", capsize=3, elinewidth=0.9, capthick=0.9)

    fig, ax = plt.subplots(figsize=(11.2, 6.3))

    ax.bar(
        x - 2 * width,
        overall_mean,
        width,
        yerr=overall_err,
        color=BAR_COLORS["overall"],
        edgecolor="white",
        label="Overall Accuracy",
        error_kw=error_style,
    )
    ax.bar(
        x - width,
        g0_mean,
        width,
        yerr=g0_err,
        color=BAR_COLORS["g0"],
        edgecolor="white",
        label=group_names[0],
        error_kw=error_style,
    )
    ax.bar(
        x,
        g1_mean,
        width,
        yerr=g1_err,
        color=BAR_COLORS["g1"],
        edgecolor="white",
        label=group_names[1],
        error_kw=error_style,
    )
    ax.bar(
        x + width,
        g2_mean,
        width,
        yerr=g2_err,
        color=BAR_COLORS["g2"],
        edgecolor="white",
        label=group_names[2],
        error_kw=error_style,
    )
    ax.bar(
        x + 2 * width,
        g3_mean,
        width,
        yerr=g3_err,
        color=BAR_COLORS["g3"],
        edgecolor="white",
        label=group_names[3],
        error_kw=error_style,
    )

    group_centers = np.array([x - width, x, x + width, x + 2 * width], dtype=float)
    for method_idx in range(len(task_df)):
        add_gap_annotation(ax, group_centers[:, method_idx], group_means, group_errors, method_idx)

    baseline_row = task_df[task_df["method_label"] == "Baseline (ERM)"].iloc[0]
    ax.axhline(
        float(baseline_row["wga_mean"]),
        color="#9b6d6d",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label="Baseline Worst Group",
    )

    max_height = max(
        np.max(overall_mean + overall_err[1]),
        np.max(g0_mean + g0_err[1]),
        np.max(g1_mean + g1_err[1]),
        np.max(g2_mean + g2_err[1]),
        np.max(g3_mean + g3_err[1]),
    )
    ax.set_ylim(0.0, min(1.16, max(1.06, max_height + 0.16)))
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("De-biasing Method", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_xtick_labels(task_df["method_label"]), fontsize=10, rotation=10)
    style_method_xticklabels(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title(
        TASK_TITLES.get(task, task) + "\n(bootstrap mean with 95% CI)",
        fontsize=15,
        weight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.96])
    out_path = OUT_DIR / TASK_OUTPUTS.get(task, f"bootstrap_{_safe_task_name(task)}_acc.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    df = load_summary(args.csv)
    tasks = [args.task] if args.task else list(TASK_TITLES)
    for task in tasks:
        out_path = plot_task(df, task)
        print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
