import torch.nn as nn
from torchvision import models


class CIFAR10_ResNet18(nn.Module):
    """ResNet18 backbone adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class CIFAR10_DenseNet121(nn.Module):
    """DenseNet121 backbone adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)
