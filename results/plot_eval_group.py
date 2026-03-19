from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
绘制任务分组准确率柱状图。
用法：
  cd D:\Code\DeepLearning\CelebA\results
  python plot_eval_group.py
"""

CSV_PATH = Path(__file__).parent / "eval_summary.csv"
OUT_DIR = Path(__file__).parent

METHOD_ORDER = [
    "Baseline (ERM)",
    "FSC (Unbalanced)",
    "FSC (Oversampling)",
    "FSC (Reweighting)",
]

TASK_TITLES = {
    "BlondHair_Male": "Blond × Male - Accuracy Breakdown by Demographic Groups",
    "MouthOpen_Smiling": "Mouth × Smiling - Accuracy Breakdown by Demographic Groups",
}


def _safe_task_name(task: str) -> str:
    return task.lower().replace(" ", "_")


def _task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = df[df["task"] == task].copy()
    task_df["method_label"] = pd.Categorical(task_df["method_label"], METHOD_ORDER, ordered=True)
    task_df = task_df.sort_values("method_label")
    if task_df.empty:
        raise ValueError(f"No rows found for task: {task}")
    return task_df


def plot_task_axes(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    *,
    title: str | None = None,
    show_legend: bool = True,
    legend_fontsize: int = 9,
) -> None:
    task_df = _task_frame(df, task)
    group_names = [task_df.iloc[0][f"group{i}_name"] for i in range(4)]
    overall = task_df["overall_acc"].to_numpy()
    g0 = task_df["acc_g0"].to_numpy()
    g1 = task_df["acc_g1"].to_numpy()
    g2 = task_df["acc_g2"].to_numpy()
    g3 = task_df["acc_g3"].to_numpy()

    x = np.arange(len(task_df))
    width = 0.16

    ax.bar(x - 2 * width, overall, width, color="#34495e", edgecolor="black", label="Overall Accuracy")
    ax.bar(x - width, g0, width, color="#5dade2", edgecolor="black", label=group_names[0])
    ax.bar(x, g1, width, color="#58d68d", edgecolor="black", label=group_names[1])
    ax.bar(x + width, g2, width, color="#ec7063", edgecolor="black", label=group_names[2])
    ax.bar(x + 2 * width, g3, width, color="#d4ac0d", edgecolor="black", label=group_names[3])

    baseline_row = task_df[task_df["method_label"] == "Baseline (ERM)"].iloc[0]
    baseline_worst = float(baseline_row["wga"])
    ax.axhline(
        y=baseline_worst,
        color="#9b6d6d",
        linestyle="--",
        linewidth=1.4,
        alpha=0.65,
        label="Baseline Worst Group",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(task_df["method_label"], fontsize=9, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlabel("De-biasing Method", fontsize=11)
    ax.set_title(title or TASK_TITLES.get(task, task), fontsize=13, weight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True,
            fontsize=legend_fontsize,
        )


def plot_task(df: pd.DataFrame, task: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_task_axes(ax, df, task, title=TASK_TITLES.get(task, task), show_legend=True, legend_fontsize=9)
    fig.tight_layout(rect=[0.02, 0.02, 0.74, 0.98])
    out_path = OUT_DIR / f"{_safe_task_name(task)}_group_accuracy_breakdown.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_blond_mouth_1x2(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
    plot_task_axes(
        axes[0],
        df,
        "BlondHair_Male",
        title=TASK_TITLES["BlondHair_Male"],
        show_legend=True,
        legend_fontsize=8,
    )
    plot_task_axes(
        axes[1],
        df,
        "MouthOpen_Smiling",
        title=TASK_TITLES["MouthOpen_Smiling"],
        show_legend=True,
        legend_fontsize=8,
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.88, 0.96], pad=1.2)
    fig.subplots_adjust(wspace=0.42)
    out_path = OUT_DIR / "eval_summary.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    out = plot_blond_mouth_1x2(df)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
