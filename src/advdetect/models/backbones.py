from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class NormalizeInput(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


class NormalizedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.normalize = NormalizeInput(mean, std)
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(images))

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits and backbone stage activations for detector baselines."""
        if not hasattr(self.backbone, "forward_features"):
            raise TypeError("The wrapped classifier does not expose forward_features")
        return self.backbone.forward_features(self.normalize(images))


class TimmResNet50Classifier(nn.Module):
    """timm ResNet50 wrapper exposing the feature map expected by LR."""

    def __init__(self, num_classes: int, normalize_input: bool,
                 input_mean: tuple[float, ...], input_std: tuple[float, ...],
                 activation: str = "relu") -> None:
        super().__init__()
        import timm
        self.normalize = NormalizeInput(input_mean, input_std) if normalize_input else nn.Identity()
        activation = activation.lower()
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation in {"leaky_relu", "leakyrelu"}:
            act_layer = lambda **kwargs: nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            raise ValueError(f"Unsupported ResNet50 activation: {activation}")
        self.activation = activation
        self.backbone = timm.create_model("resnet50", pretrained=False,
                                          num_classes=num_classes, act_layer=act_layer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(self.normalize(images))
        return self.backbone.forward_head(features)

    def forward_features(self, images: torch.Tensor):
        features = self.backbone.forward_features(self.normalize(images))
        logits = self.backbone.forward_head(features)
        return logits, [features]


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class CIFARResNet18(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._layer(64, 64, 2, 1)
        self.layer2 = self._layer(64, 128, 2, 2)
        self.layer3 = self._layer(128, 256, 2, 2)
        self.layer4 = self._layer(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [CIFARBasicBlock(in_channels, out_channels, stride)]
        layers.extend(CIFARBasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits and post-stage feature maps, matching Mahalanobis/LID layers."""
        x = self.relu(self.bn1(self.conv1(x)))
        features = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            features.append(x)
        pooled = self.pool(features[-1]).flatten(1)
        return self.fc(pooled), features


def build_classifier(
    arch: str,
    in_channels: int,
    num_classes: int,
    small_input: bool = True,
    normalize_input: bool = False,
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
    activation: str = "relu",
) -> nn.Module:
    normalized = arch.lower()
    handles_normalization = False
    if normalized == "cifar_resnet18":
        model: nn.Module = CIFARResNet18(in_channels, num_classes)
    elif normalized in {"timm_resnet50", "resnet50_timm"}:
        if input_mean is None or input_std is None:
            raise ValueError("input_mean and input_std are required for timm_resnet50")
        model = TimmResNet50Classifier(num_classes, normalize_input, input_mean, input_std,
                                       activation=activation)
        handles_normalization = True
    else:
        if normalized == "resnet18":
            model = resnet18(weights=None)
        elif normalized == "resnet50":
            model = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported classifier architecture: {arch}")
        original = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            original.out_channels,
            3 if small_input else original.kernel_size,
            1 if small_input else original.stride,
            1 if small_input else original.padding,
            bias=False,
        )
        if small_input:
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    if normalize_input and not handles_normalization:
        if input_mean is None or input_std is None:
            raise ValueError("input_mean and input_std are required for input normalization")
        model = NormalizedClassifier(model, input_mean, input_std)
    return model
