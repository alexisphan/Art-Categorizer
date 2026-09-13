"""
Backbone + classification head. Same architecture reused across all three
tasks (only the head's output size differs, since num_classes differs).
"""

import torch.nn as nn
from torchvision import models


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
        return net

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
        return net

    if backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        return net

    raise ValueError(
        f"Unknown backbone '{backbone}'. Supported: resnet18, resnet50, efficientnet_b0. "
        f"Add more in src/model.py if needed."
    )