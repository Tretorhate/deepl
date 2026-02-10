"""
Vision Models for Medical Image Classification
Implements ResNet and other CNN architectures for medical imaging
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """
    ResNet-based image classifier for medical images.

    Supports ResNet-18 and ResNet-50 with pretrained ImageNet weights.
    """

    def __init__(self, num_classes=2, architecture='resnet18', pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            architecture: 'resnet18' or 'resnet50'
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate for regularization
        """
        super(ResNetClassifier, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Load pretrained model
        if architecture == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
        elif architecture == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Replace classification head
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features // 2, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 3, height, width)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all layers except the classification head."""
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = False
        print(f"Froze backbone for {self.architecture}")

    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"Unfroze backbone for {self.architecture}")


class SimpleConvNet(nn.Module):
    """
    Simple CNN for medical image classification.

    Useful baseline model for testing and when computational resources are limited.
    """

    def __init__(self, num_classes=2, dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SimpleConvNet, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 3, height, width)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    print("Testing vision models...")

    # Test ResNet18
    print("\nTesting ResNet18...")
    resnet18 = ResNetClassifier(num_classes=2, architecture='resnet18', pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    output = resnet18(x)
    print(f"ResNet18 output shape: {output.shape}")
    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

    # Test ResNet50
    print("\nTesting ResNet50...")
    resnet50 = ResNetClassifier(num_classes=2, architecture='resnet50', pretrained=False)
    output = resnet50(x)
    print(f"ResNet50 output shape: {output.shape}")
    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

    # Test SimpleConvNet
    print("\nTesting SimpleConvNet...")
    simple_cnn = SimpleConvNet(num_classes=2)
    output = simple_cnn(x)
    print(f"SimpleConvNet output shape: {output.shape}")
    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

    # Test backbone freezing
    print("\nTesting backbone freezing...")
    resnet18.freeze_backbone()
    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params}")

    resnet18.unfreeze_backbone()
    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {trainable_params}")

    print("\nAll vision model tests passed!")
