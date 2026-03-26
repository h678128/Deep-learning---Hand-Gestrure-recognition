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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HandLandmarkCNN(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 6,
        output_dims: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.output_dims = output_dims

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_landmarks * output_dims),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        coordinates = self.regressor(features)
        return coordinates.view(-1, self.num_landmarks, self.output_dims)


def create_landmark_model(
    num_landmarks: int = 6,
    output_dims: int = 2,
    dropout: float = 0.3,
) -> HandLandmarkCNN:
    return HandLandmarkCNN(
        num_landmarks=num_landmarks,
        output_dims=output_dims,
        dropout=dropout,
    )
