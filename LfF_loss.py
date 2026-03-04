"""
LfF 损失函数：
  - GeneralizedCELoss: 用于训练有偏模型（Stage 1），使其聚焦于简单/偏见对齐样本
  - WeightedCELoss:    用于训练主模型（Stage 3），根据样本权重加权 CE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCELoss(nn.Module):
    """
    Generalized Cross Entropy (GCE) 损失。

    公式:  L_GCE(p, y) = (1 - p_y^q) / q
      p_y = softmax(logits)_y 为真实类别的预测概率

    性质: 较大的 q 会截断高损失样本的梯度，使优化被"简单样本"主导。
    在存在偏见的数据上，简单样本就是偏见对齐的样本
    （如 NonBlond Male、Blond Female），
    因此 GCE 训练出的模型会强化偏见 —— 这正是 LfF 第一阶段所需要的。

    特殊情况: q→0 退化为标准 CE；q=1 退化为 MAE。
    """

    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        p_y = p.gather(1, targets.view(-1, 1)).squeeze(1)
        p_y = torch.clamp(p_y, min=1e-7, max=1.0)
        loss = (1.0 - p_y ** self.q) / self.q
        return loss.mean()


class WeightedCELoss(nn.Module):
    """
    逐样本加权交叉熵损失。

    每个样本的 CE 损失乘以其权重（来自有偏模型的失败分析）。
    有偏模型预测失败的样本（偏见冲突样本，如 Blond Male）获得更高权重，
    迫使主模型重点学习这些被偏见忽略的样本。
    """

    def forward(self, logits, targets, weights):
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
        return (per_sample_loss * weights).mean()
