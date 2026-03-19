"""
Experimental Attribute Bias Heatmap
=====================================
生成一张专注于实验用属性对的 bias 热力图：
  - 行 = 候选 Target 属性（你想预测的）
  - 列 = 候选 Sensitive 属性（可能产生 bias 的）
  - 颜色 = DPD（在子集内归一化），越深代表 bias 越大
  - 每格标注：DPD 值 + 最小组样本数

用法：
  cd D:/Code/DeepLearning/CelebA/bias_analysis
  python plot_bias_heatmap.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# ─── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
ATTR_CSV = ROOT / "datasets/list_attr_celeba.csv"
OUT_DIR = ROOT / "bias_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 实验候选属性（可自由增删）─────────────────────────────────────────────────
TARGET_ATTRS = [
    # ── 性别相关 target ──
    "Wearing_Lipstick",
    "Heavy_Makeup",
    "Arched_Eyebrows",
    "Wearing_Necktie",
    "Blond_Hair",
    # ── 外貌 / 表情 target ──
    "Attractive",
    "Smiling",
    "Mouth_Slightly_Open",
    "High_Cheekbones",
    "Wavy_Hair",
    # ── 年龄相关 target ──
    "Young",
    "Receding_Hairline",
    # ── 经典对照 target ──
    "Eyeglasses",
    "Chubby",
    "Double_Chin",
]

SENSITIVE_ATTRS = [
    "Male",
    "Young",
    "High_Cheekbones",
    "Smiling",
    "Wearing_Lipstick",
    "Heavy_Makeup",
    "Gray_Hair",
    "Chubby",
    "Double_Chin",
    "Rosy_Cheeks",
]

# 高亮条件：DPD >= 此值 且 min_group >= MIN_GROUP_THRESHOLD
HIGHLIGHT_DPD_THRESHOLD = 0.40
MIN_GROUP_THRESHOLD     = 1000

# 已选定用于 ResNet-18 实验的 pair（橙色粗框标注）
SELECTED_PAIRS = {
    ("Blond_Hair",         "Male"),
    ("Mouth_Slightly_Open","Smiling"),
}


# ─── 计算函数 ──────────────────────────────────────────────────────────────────
def load_binary(csv_path: Path):
    df = pd.read_csv(csv_path)
    attr_cols = [c for c in df.columns if c != "image_id"]
    df[attr_cols] = ((df[attr_cols] + 1) // 2).astype(np.int8)
    return df, attr_cols


def compute_cell(T: np.ndarray, S: np.ndarray):
    """返回 (dpd, min_group_count)。"""
    s1 = S == 1
    s0 = S == 0
    p1 = T[s1].mean() if s1.any() else 0.0
    p0 = T[s0].mean() if s0.any() else 0.0
    dpd = abs(p1 - p0)

    g0 = int(((T == 0) & (S == 0)).sum())
    g1 = int(((T == 0) & (S == 1)).sum())
    g2 = int(((T == 1) & (S == 0)).sum())
    g3 = int(((T == 1) & (S == 1)).sum())
    min_g = min(g0, g1, g2, g3)
    return round(dpd, 3), min_g


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {ATTR_CSV} ...")
    df, _ = load_binary(ATTR_CSV)

    # 过滤掉 target == sensitive 的情况
    tgts = [a for a in TARGET_ATTRS  if a in df.columns]
    sens = [a for a in SENSITIVE_ATTRS if a in df.columns]

    nT, nS = len(tgts), len(sens)
    dpd_mat  = np.zeros((nT, nS))
    ming_mat = np.zeros((nT, nS), dtype=int)

    for i, t in enumerate(tgts):
        for j, s in enumerate(sens):
            if t == s:
                dpd_mat[i, j]  = np.nan
                ming_mat[i, j] = 0
            else:
                dpd_mat[i, j], ming_mat[i, j] = compute_cell(
                    df[t].values, df[s].values
                )

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(nS * 1.55 + 1.5, nT * 1.3 + 1.5))

    # 颜色在子集内归一化（忽略 NaN）
    vmax = float(np.nanmax(dpd_mat))
    cmap = plt.cm.YlOrRd

    # 先铺底色
    im = ax.imshow(dpd_mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    # 每格写文字 + 高亮边框
    for i, t in enumerate(tgts):
        for j, s in enumerate(sens):
            if np.isnan(dpd_mat[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    color="#e0e0e0", zorder=1))
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#aaaaaa")
                continue

            dpd  = dpd_mat[i, j]
            ming = ming_mat[i, j]
            # 字色随背景亮度自动切换
            brightness = dpd / vmax if vmax > 0 else 0
            txt_color  = "white" if brightness > 0.55 else "black"

            ax.text(j, i - 0.12, f"{dpd:.3f}",
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=txt_color)
            ax.text(j, i + 0.27, f"min={ming:,}",
                    ha="center", va="center", fontsize=7.5,
                    color=txt_color, alpha=0.85)

            # 已选定实验对 → 橙色粗框（优先级最高，覆盖其他边框）
            # DPD 高且 min 充足 → 绿色实线边框
            # DPD 高但 min 不足 → 灰色虚线边框
            if (t, s) in SELECTED_PAIRS:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=10.0, 
                    edgecolor="#1a6ee8",
                    facecolor="none", zorder=4,
                    linestyle="solid")
                ax.add_patch(rect)
            elif dpd >= HIGHLIGHT_DPD_THRESHOLD:
                if ming >= MIN_GROUP_THRESHOLD:
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=4, edgecolor="#2ca02c",
                        facecolor="none", zorder=3,
                        linestyle="solid")
                else:
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=3, edgecolor="#888888",
                        facecolor="none", zorder=3,
                        linestyle="dashed")
                ax.add_patch(rect)

    # 轴标签
    ax.set_xticks(range(nS))
    ax.set_yticks(range(nT))
    ax.set_xticklabels(sens, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(tgts, fontsize=10)
    ax.set_xlabel("Sensitive Attribute", fontsize=11, labelpad=8)
    ax.set_ylabel("Target Attribute", fontsize=11, labelpad=8)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("DPD (Demographic Parity Difference)", fontsize=9)

    # 图例
    legend_selected = mpatches.Patch(
        facecolor="none", edgecolor="#1a6ee8", linewidth=3.0,
        label="Selected for ResNet-18 experiment")
    legend_good = mpatches.Patch(
        facecolor="none", edgecolor="#2ca02c", linewidth=3.0,
        label=f"Recommended (DPD >= {HIGHLIGHT_DPD_THRESHOLD} & min >= {MIN_GROUP_THRESHOLD:,})")
    legend_warn = mpatches.Patch(
        facecolor="none", edgecolor="#888888", linewidth=1.8,
        linestyle="dashed",
        label=f"High DPD but small min (< {MIN_GROUP_THRESHOLD:,})")
    ax.legend(handles=[legend_selected, legend_good, legend_warn],
              loc="upper right", fontsize=8, framealpha=0.85)

    ax.set_title(
        "CelebA Attribute Pair Bias — DPD Heatmap\n"
        "blue = selected experiment pairs; "
        f"green = recommended (DPD >= {HIGHLIGHT_DPD_THRESHOLD} & min >= {MIN_GROUP_THRESHOLD:,}); "
        "grey dashed = sparse",
        fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    out = OUT_DIR / "bias_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 控制台汇总 ────────────────────────────────────────────────────────────
    print(f"\nRecommended pairs (DPD >= {HIGHLIGHT_DPD_THRESHOLD} & min >= {MIN_GROUP_THRESHOLD:,}), sorted by DPD:")
    print(f"  {'Target':<22} {'Sensitive':<22} {'DPD':>6}  {'MinGroup':>9}")
    print("  " + "-" * 60)
    good, warn = [], []
    for i, t in enumerate(tgts):
        for j, s in enumerate(sens):
            if np.isnan(dpd_mat[i, j]) or dpd_mat[i, j] < HIGHLIGHT_DPD_THRESHOLD:
                continue
            entry = (dpd_mat[i, j], ming_mat[i, j], t, s)
            if ming_mat[i, j] >= MIN_GROUP_THRESHOLD:
                good.append(entry)
            else:
                warn.append(entry)
    for dpd, ming, t, s in sorted(good, reverse=True):
        print(f"  {t:<22} {s:<22} {dpd:>6.3f}  {ming:>9,}")
    if warn:
        print(f"\nHigh DPD but sparse (min < {MIN_GROUP_THRESHOLD:,}):")
        for dpd, ming, t, s in sorted(warn, reverse=True):
            print(f"  {t:<22} {s:<22} {dpd:>6.3f}  {ming:>9,}  [sparse]")

    print("\nSelected ResNet-18 experiment pairs:")
    print(f"  {'Target':<22} {'Sensitive':<22} {'DPD':>6}  {'MinGroup':>9}")
    print("  " + "-" * 60)
    tgt_idx = {t: i for i, t in enumerate(tgts)}
    sen_idx = {s: j for j, s in enumerate(sens)}
    for t, s in sorted(SELECTED_PAIRS):
        if t in tgt_idx and s in sen_idx:
            i, j = tgt_idx[t], sen_idx[s]
            dpd  = dpd_mat[i, j]
            ming = ming_mat[i, j]
            flag = "" if ming >= MIN_GROUP_THRESHOLD else "  [sparse!]"
            print(f"  {t:<22} {s:<22} {dpd:>6.3f}  {ming:>9,}{flag}")


if __name__ == "__main__":
    main()
