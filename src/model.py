from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleHandHeatmapCNN(nn.Module):
    def __init__(self, num_landmarks: int = 11) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks

        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(256, 128)
        self.output_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec1 = self.up1(bottleneck)
        dec1 = self.decoder1(torch.cat([dec1, enc3], dim=1))
        return self.output_head(dec1)


class HandHeatmapCNN(nn.Module):
    def __init__(self, num_landmarks: int = 11) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks

        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.encoder4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.output_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec1 = self.up1(bottleneck)
        dec1 = self.decoder1(torch.cat([dec1, enc4], dim=1))
        dec2 = self.up2(dec1)
        dec2 = self.decoder2(torch.cat([dec2, enc3], dim=1))
        return self.output_head(dec2)


def infer_heatmap_model_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("encoder4.") for key in state_dict):
        return "deep"
    return "simple"


def create_heatmap_model(
    num_landmarks: int = 11,
    architecture: str = "deep",
) -> nn.Module:
    if architecture == "deep":
        return HandHeatmapCNN(num_landmarks=num_landmarks)
    if architecture == "simple":
        return SimpleHandHeatmapCNN(num_landmarks=num_landmarks)
    raise ValueError(f"Unsupported heatmap model architecture: {architecture}")


def create_heatmap_model_from_checkpoint(checkpoint: dict[str, object]) -> nn.Module:
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict.")

    architecture = str(
        checkpoint.get("model_architecture")
        or infer_heatmap_model_architecture(state_dict)
    )
    num_landmarks = int(checkpoint.get("num_landmarks", 11))
    return create_heatmap_model(
        num_landmarks=num_landmarks,
        architecture=architecture,
    )
