"""
LfF 分类模型：标准 ResNet-18 单头分类器。

与 Fair Contrastive Learning 的双头模型不同，LfF 不需要投影头，
仅使用分类头。有偏模型和主模型共用相同架构，仅训练方式不同。
"""
import torch.nn as nn
from torchvision import models


class LfFClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)
