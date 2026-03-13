import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class FairSupConLoss(nn.Module):
    # 有 sensitives 时：正对 = same target & different sensitive（跨组对比，迫使表征忽略 sensitive 信息）。
    # 无 sensitives 时：正对 = same target（普通 SupCon，用于 baseline 对比）。
    """基于 SupConLoss (Khosla et al., NeurIPS 2020) 的公平对比损失。
    参考: https://arxiv.org/abs/2004.11362

    ── 公式（与 README §2.2 FairSupCon Loss 一致）────────────────────────────
    记: anchor 为 i, batch B, 标签 y, 敏感属性 s, 嵌入 z (L2 归一化), 温度 τ。

    (1) 公平正样本集（只与“同标签、不同敏感属性”的样本拉近）：
        P_Fair(i) = { j : j ≠ i,  y_j = y_i,  s_j ≠ s_i }
    (2) 负样本集（不同标签）：
        N(i) = { k : y_k ≠ y_i }
    (3) 分母集合（参与 softmax 的样本：公平正样本 ∪ 负样本；同标签同敏感属性既不拉近也不推远）：
        D(i) = P_Fair(i) ∪ N(i)
    (4) 相似度（余弦，因 z 已 L2 归一化）：
        sim(i, j) = z_i · z_j
    (5) FairSupCon 损失：
        L_FSC = - (1/|B|)  Σ_{i∈B} (1/|P_Fair(i)|)  Σ_{j∈P_Fair(i)} log [exp(sim(i,j)/τ) / Σ_{k∈D(i)} exp(sim(i,k)/τ)]

    无 sensitives 时退化为普通 SupCon: P(i) = 同标签非自身, D(i) = 除自身外全体。

    相比原版 SupConLoss 的改动：
    ┌───────────────┬──────────────────────────┬──────────────────────────────────┐
    │  改动点        │  原版 SupConLoss         │  FairSupConLoss                  │
    ├───────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 1. 输入形状    │ [B, n_views, feat_dim]   │ [B, feat_dim] 单视图             │
    │               │ 支持多视图 (two crops)    │ 用于 CE + SupCon 联合训练         │
    ├───────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 2. 正样本定义  │ 同 label 即正样本         │ 有 sensitives: 同 label 且不同    │
    │               │ mask = eq(labels,labels.T)│   sensitive(跨组对比, 促使表征    │
    │               │                          │   忽略敏感属性）                   │
    │               │                          │ 无 sensitives: 退化为同 label     │
    ├───────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 3. 分母构造    │ 除自身外所有样本          │ 有 sensitives: 额外排除"同 label  │
    │               │ denom = logits_mask      │   且同 sensitive"的样本           │
    │               │                          │   D(i) = P_Fair(i) ∪ N(i)       │
    │               │                          │ 无 sensitives: 同原版             │
    ├───────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 4. 温度缩放    │ loss *= temperature /    │ 去掉 base_temperature，直接取负   │
    │               │         base_temperature │   均值                           │
    ├───────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 5. 其他简化    │ contrast_mode, 自定义mask│ 去掉；新增 B<=1 边界保护           │
    └───────────────┴──────────────────────────┴──────────────────────────────────┘
    """

    def __init__(self, temperature=cfg.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, sensitives=None):
        """
        Args:
            features:    [B, feat_dim]，L2 归一化的嵌入（改动：原版为 [B, n_views, feat_dim] 多视图）
            labels:      [B]，目标标签。
            sensitives:  [B] 或 None，敏感属性.（改动：原版无此参数）
        Returns:
            scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # 改动：原版无此保护，这里对极小 batch 直接返回 0
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        # mask: 同 label 的样本对标记为 1（与原版一致）
        # 公式 (1)(2)：同标签矩阵 → 用于构造 P_Fair(i) 与 N(i)
        # mask[i,j]=1 ⇔ y_i=y_j
        mask = torch.eq(labels, labels.T).float().to(device)

        # 公式 (3)(5)：排除自身（j≠i, k≠i）
        # logits_mask[i,k]=0 当且仅当 k=i
        logits_mask = 1.0 - torch.eye(batch_size, device=device)

        if sensitives is not None:
            sensitives = sensitives.contiguous().view(-1, 1)
            diff_sensitive = torch.ne(sensitives, sensitives.T).float().to(device)

            # 改动: 正样本 = 同 label ∩ 不同 sensitive ∩ 非自身（原版：正样本 = 同 label ∩ 非自身）
            # 公式 (1)：P_Fair(i) 的 mask
            # mask_pos[i,j]=1 ⇔ j∈P_Fair(i)，即 j≠i, y_j=y_i, s_j≠s_i
            mask_pos = mask * diff_sensitive * logits_mask

            # 分母排除"同 label ∩ 同 sensitive"，即 D(i) = P_Fair(i) ∪ N(i)
            # 公式 (3)：D(i) = P_Fair(i) ∪ N(i) 的 mask
            # 分母中排除“同标签且同敏感属性”的样本，只保留 P_Fair ∪ N
            same_label_same_sens = mask * (1.0 - diff_sensitive) * logits_mask
            denom_mask = logits_mask - same_label_same_sens
        else:
            # 无 sensitives 时退化为原版 SupConLoss 行为: P(i)=同标签非自身，D(i)=除自身外全体
            mask_pos = mask * logits_mask
            denom_mask = logits_mask

        # 计算缩放相似度
        # 公式 (4)(5)：sim(i,k)/τ，并做数值稳定（减行最大值）
        # anchor_dot_contrast[i,k] = z_i·z_k/τ
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 改动：exp_logits 乘 denom_mask 而非 logits_mask，排除同组同标签样本
        # 公式 (5)：log [ exp(sim(i,j)/τ) / Σ_{k∈D(i)} exp(sim(i,k)/τ) ]
        # 分母求和只对 k∈D(i)，即 exp_logits * denom_mask 再 sum
        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 正样本平均 log-prob（与原版结构一致）
        # 公式 (5)：对 j∈P_Fair(i) 求平均 → (1/|P_Fair(i)|) Σ_{j∈P_Fair(i)} log(...)
        mask_pos_pairs = mask_pos.sum(1)  # |P_Fair(i)|
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos_pairs + 1e-8)

        # 改动：原版对全部样本取 mean 并乘 temperature/base_temperature
        #       这里只对有正样本的样本取 mean，去掉了 base_temperature 缩放，直接取负均值
        # 公式 (5)：L_FSC = - (1/|B|) Σ_{i∈B} mean_log_prob_pos(i)，仅对 |P_Fair(i)|>0 的 i 平均
        loss = -mean_log_prob_pos[mask_pos_pairs > 0].mean()
        return loss


class GroupWeightedCrossEntropyLoss(nn.Module):
    """Group-weighted CE loss used by reweighting mode."""

    def __init__(self, group_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("group_weights", group_weights.float())

    def forward(self, logits, targets, sensitives):
        groups = targets * 2 + sensitives
        weights = self.group_weights[groups]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        return (ce_loss * weights).mean()


class TotalLoss(nn.Module):
    """总损失（README §2.3）：L_total = L_CE + λ·L_FSC。"""

    def __init__(self, lambda_con=cfg.LAMBDA_CON, temperature=cfg.TEMPERATURE, group_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss() if group_weights is None else GroupWeightedCrossEntropyLoss(group_weights)
        self.supcon = FairSupConLoss(temperature=temperature)
        self.lam = lambda_con
        self.use_group_reweight = group_weights is not None

    def forward(self, logits, embeddings, targets, sensitives=None):
        if self.use_group_reweight:
            if sensitives is None:
                raise ValueError("sensitives is required when group reweighting is enabled")
            ce = self.ce(logits, targets, sensitives)
        else:
            ce = self.ce(logits, targets)
        con = self.supcon(embeddings, targets, sensitives)
        return ce + self.lam * con, ce, con
