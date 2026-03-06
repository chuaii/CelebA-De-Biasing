import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config


class FairClassifier(nn.Module):
    """ResNet-18 → projection head (embedding) + classification head (logits)。"""

    def __init__(self, embed_dim=config.EMBED_DIM, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, embed_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        embedding = F.normalize(self.projector(feat))
        logits = self.classifier(feat)
        return embedding, logits
