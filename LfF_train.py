"""
Learning from Failure (LfF) — CelebA 去偏训练流程

三阶段管线：
  Stage 1  训练有偏模型    使用 GCE 损失使模型故意过拟合数据偏见
  Stage 2  失败识别与权重计算  用有偏模型的 CE 损失给每个训练样本赋权
  Stage 3  训练去偏主模型    在加权 CE 损失下训练全新 ResNet-18

核心思想：有偏模型擅长识别"简单"（偏见对齐）样本，
因此它失败的样本就是偏见冲突样本。通过提升这些样本的权重，
主模型被迫学习到不依赖偏见的决策边界。
"""
import argparse
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

import LfF_config as cfg
from LfF_model import LfFClassifier
from LfF_loss import GeneralizedCELoss, WeightedCELoss
from LfF_dataset import get_train_loader, get_weight_loader, get_eval_loader
from LfF_eval import lff_evaluate, print_metrics


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ======================================================================
#  Stage 1: Train Biased Model
# ======================================================================

def train_biased_model(train_loader, device, epochs, gce_q, lr):
    """
    用 GCE 损失训练有偏模型。

    GCE 会截断难样本（偏见冲突样本）的梯度，使模型的优化被"简单"样本
    （偏见对齐样本）主导。经过少量 epoch 训练，模型学到的决策规则近似于：
      "不是男性 → 可能是金发" （偏见捷径）
    而非真正基于发色特征做判断。
    """
    model = LfFClassifier().to(device)
    criterion = GeneralizedCELoss(q=gce_q)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets, _, _, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            loss = criterion(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  [Biased] Epoch {ep + 1}/{epochs}  "
              f"GCE_loss={total_loss / n:.4f}")

    return model


# ======================================================================
#  Stage 2: Failure Identification & Re-weighting
# ======================================================================

@torch.no_grad()
def compute_sample_weights(biased_model, weight_loader, device):
    """
    用有偏模型的标准 CE 损失衡量每个训练样本的"难度"。

    有偏模型预测失败 → 高 CE 损失 → 偏见冲突样本 → 赋予高权重。
    有偏模型预测正确 → 低 CE 损失 → 偏见对齐样本 → 赋予低权重。

    归一化方式：除以全局均值损失，使平均权重 ≈ 1，
    再 clamp 到 [WEIGHT_CLAMP_MIN, WEIGHT_CLAMP_MAX] 防止极端值。
    """
    biased_model.eval()
    num_samples = len(weight_loader.dataset)
    losses = torch.zeros(num_samples)

    for images, targets, _, _, indices in weight_loader:
        images, targets = images.to(device), targets.to(device)
        logits = biased_model(images)
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
        losses[indices] = per_sample_loss.cpu()

    mean_loss = losses.mean()
    weights = losses / (mean_loss + 1e-8)
    weights = torch.clamp(weights, min=cfg.WEIGHT_CLAMP_MIN, max=cfg.WEIGHT_CLAMP_MAX)

    print(f"  Loss stats: mean={mean_loss:.4f}  std={losses.std():.4f}  "
          f"min={losses.min():.4f}  max={losses.max():.4f}")
    print(f"  Weight stats: mean={weights.mean():.3f}  "
          f"min={weights.min():.3f}  max={weights.max():.3f}")

    return weights


def analyze_weights(weights, dataset):
    """打印各组的权重统计，验证偏见冲突组（Blond_Male）获得了最高权重。"""
    groups = dataset.groups
    group_weights = {g: [] for g in range(4)}
    for i, g in enumerate(groups):
        group_weights[g].append(weights[i].item())

    print("\n  [Weight Distribution by Group]")
    for g in range(4):
        w = torch.tensor(group_weights[g])
        print(f"    {cfg.GROUP_NAMES[g]:20s}:  mean={w.mean():.3f}  "
              f"median={w.median():.3f}  "
              f"min={w.min():.3f}  max={w.max():.3f}  n={len(w)}")
    print()


# ======================================================================
#  Stage 3: Train Main / Debiased Model
# ======================================================================

def train_main_model(train_loader, val_loader, sample_weights, device, args):
    """
    用加权 CE 损失训练全新 ResNet-18。

    高权重样本（偏见冲突样本，如 Blond Male）在损失中获得更大贡献，
    迫使模型认真学习这些被有偏模型忽略的样本，从而建立
    基于发色（而非性别）的决策边界。
    """
    model = LfFClassifier().to(device)
    criterion = WeightedCELoss()

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.MAIN_LR_BACKBONE},
        {"params": head_params, "lr": args.main_lr},
    ], weight_decay=cfg.MAIN_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.main_epochs
    )

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    best_wga, no_improve = 0.0, 0
    best_path = os.path.join(cfg.CKPT_DIR, "best_LfF_main.pt")

    for ep in range(args.main_epochs):
        t0 = time.time()
        model.train()
        total_loss, n = 0.0, 0

        for images, targets, _, _, indices in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            w = sample_weights[indices].to(device)

            loss = criterion(model(images), targets, w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        scheduler.step()

        m = lff_evaluate(model, val_loader, device)
        wga = m["worst_group_acc"]
        elapsed = time.time() - t0

        print(f"  [Main] Epoch {ep + 1}/{args.main_epochs}  {elapsed:.0f}s  "
              f"loss={total_loss / n:.4f}  acc={m['overall_acc']:.2%}  "
              f"wga={wga:.2%}  unbiased={m['unbiased_acc']:.2%}")
        for g in range(4):
            print(f"    {cfg.GROUP_NAMES[g]}: {m['group_acc'][g]:.2%}")

        if wga > best_wga:
            best_wga = wga
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"    -> saved {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {ep + 1}, "
                      f"no WGA improvement for {args.patience} epochs")
                break

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    print(f"  Best validation WGA = {best_wga:.2%}")
    return model


# ======================================================================
#  Main
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="LfF: Learning from Failure — CelebA 去偏训练")
    p.add_argument("--biased_epochs", type=int, default=cfg.BIASED_EPOCHS, help="Stage 1 有偏模型训练轮数")
    p.add_argument("--gce_q", type=float, default=cfg.GCE_Q, help="GCE 截断参数 q")
    p.add_argument("--biased_lr", type=float, default=cfg.BIASED_LR, help="Stage 1 学习率")
    p.add_argument("--main_epochs", type=int, default=cfg.MAIN_EPOCHS, help="Stage 3 主模型最大训练轮数")
    p.add_argument("--main_lr", type=float, default=cfg.MAIN_LR, help="Stage 3 head 学习率")
    p.add_argument("--bs", type=int, default=cfg.BIASED_BATCH_SIZE, help="batch size")
    p.add_argument("--patience", type=int, default=cfg.MAIN_PATIENCE, help="early stopping patience")
    args = p.parse_args()

    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  LfF: Learning from Failure — CelebA Debiasing")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Biased : epochs={args.biased_epochs}, q={args.gce_q}, "
          f"lr={args.biased_lr}")
    print(f"  Main   : epochs={args.main_epochs}, lr={args.main_lr}, "
          f"patience={args.patience}")
    print()

    # -------------------- Load Data --------------------
    print("Loading data...")
    train_loader = get_train_loader(args.bs)
    weight_loader = get_weight_loader(args.bs)
    val_loader = get_eval_loader("val", args.bs)

    # ============ Stage 1: Train Biased Model ============
    print()
    print("=" * 60)
    print("  Stage 1: Training Biased Model (GCE Loss)")
    print("=" * 60)

    biased_model = train_biased_model(
        train_loader, device,
        args.biased_epochs, args.gce_q, args.biased_lr,
    )

    biased_path = os.path.join(cfg.CKPT_DIR, "LfF_biased.pt")
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    torch.save(biased_model.state_dict(), biased_path)
    print(f"  -> saved {biased_path}")

    m_biased_val = lff_evaluate(biased_model, val_loader, device)
    print_metrics("Biased Model (Val)", m_biased_val)

    # ============ Stage 2: Compute Sample Weights ============
    print("=" * 60)
    print("  Stage 2: Failure Identification & Re-weighting")
    print("=" * 60)

    sample_weights = compute_sample_weights(biased_model, weight_loader, device)
    analyze_weights(sample_weights, weight_loader.dataset)

    # ============ Stage 3: Train Main Model ============
    print("=" * 60)
    print("  Stage 3: Training Main Model (Weighted CE Loss)")
    print("=" * 60)

    main_model = train_main_model(
        train_loader, val_loader, sample_weights, device, args,
    )

    # ============ Final Evaluation ============
    print()
    print("=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    test_loader = get_eval_loader("test", args.bs)

    m_main_test = lff_evaluate(main_model, test_loader, device)
    print_metrics("LfF Debiased Model (Test)", m_main_test)

    m_biased_test = lff_evaluate(biased_model, test_loader, device)
    print_metrics("Biased Model (Test) — for comparison", m_biased_test)

    # ============ Summary ============
    print("=" * 60)
    print("  Summary: Biased vs Debiased")
    print("=" * 60)
    print(f"                    {'Biased':>10s}  {'LfF Main':>10s}  {'Delta':>10s}")
    print(f"  Overall Acc     : "
          f"{m_biased_test['overall_acc']:>9.2%}  "
          f"{m_main_test['overall_acc']:>9.2%}  "
          f"{m_main_test['overall_acc'] - m_biased_test['overall_acc']:>+9.2%}")
    print(f"  Worst-Group Acc : "
          f"{m_biased_test['worst_group_acc']:>9.2%}  "
          f"{m_main_test['worst_group_acc']:>9.2%}  "
          f"{m_main_test['worst_group_acc'] - m_biased_test['worst_group_acc']:>+9.2%}")
    print(f"  Unbiased Acc    : "
          f"{m_biased_test['unbiased_acc']:>9.2%}  "
          f"{m_main_test['unbiased_acc']:>9.2%}  "
          f"{m_main_test['unbiased_acc'] - m_biased_test['unbiased_acc']:>+9.2%}")
    for g in range(4):
        print(f"  {cfg.GROUP_NAMES[g]:18s}: "
              f"{m_biased_test['group_acc'][g]:>9.2%}  "
              f"{m_main_test['group_acc'][g]:>9.2%}  "
              f"{m_main_test['group_acc'][g] - m_biased_test['group_acc'][g]:>+9.2%}")
    print()


if __name__ == "__main__":
    main()
