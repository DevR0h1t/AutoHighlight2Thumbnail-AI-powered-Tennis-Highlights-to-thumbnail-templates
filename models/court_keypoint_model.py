import torch
import torch.nn as nn
import torchvision.models as models

class CourtKeypointModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, 28)

    def forward(self, x):
        return self.backbone(x)