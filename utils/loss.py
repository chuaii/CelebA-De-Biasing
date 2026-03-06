import torch
import torch.nn as nn
import config


class FairSupConLoss(nn.Module):
    # 有 sensitives 时：正对 = same target & different sensitive（跨组对比，迫使表征忽略 sensitive 信息）。
    # 无 sensitives 时：正对 = same target（普通 SupCon，用于 baseline 对比）。
    
    def __init__(self, temperature=config.TEMPERATURE):
        super().__init__()
        self.t = temperature

    def forward(self, embeddings, targets, sensitives=None):
        B = embeddings.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=embeddings.device)

        sim = embeddings @ embeddings.T / self.t
        non_self = 1.0 - torch.eye(B, device=embeddings.device)

        targets = targets.view(-1, 1)
        same_target = (targets == targets.T).float()

        if sensitives is not None:
            sensitives = sensitives.view(-1, 1)
            diff_sens = (sensitives != sensitives.T).float()
            pos_mask = same_target * diff_sens * non_self
            # D(i) = P_Fair(i) ∪ N(i); exclude same-target-same-sensitive
            denom_mask = non_self * (1.0 - same_target * (1.0 - diff_sens))
        else:
            pos_mask = same_target * non_self
            denom_mask = non_self

        sim = sim - sim.detach().max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim) * denom_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        num_pos = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)
        return -mean_log_prob[num_pos > 0].mean()


class TotalLoss(nn.Module):
    """CE + lambda * FairSupCon。"""

    def __init__(self, lambda_con=config.LAMBDA_CON):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.supcon = FairSupConLoss()
        self.lam = lambda_con

    def forward(self, logits, embeddings, targets, sensitives=None):
        ce = self.ce(logits, targets)
        con = self.supcon(embeddings, targets, sensitives)
        return ce + self.lam * con, ce, con
