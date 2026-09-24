"""EEGNet-8,2 adapted to 60-channel, 256-sample T-MSPD EEG trials.

The architecture follows the compact temporal--depthwise spatial--separable
convolution design of Lawhern et al., J Neural Eng 2018. It is trained from
scratch and never consumes EMG, audio, or an upstream checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _same_pad_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """TensorFlow-style SAME padding for a stride-one temporal convolution."""
    total = kernel_size - 1
    left = total // 2
    right = total - left
    return F.pad(x, (left, right, 0, 0))


class EEGNet82(nn.Module):
    """EEGNet-8,2 for input shaped ``(batch, 60, 256)``.

    Fixed architecture:
      F1=8, D=2, F2=16, temporal kernel=64, separable kernel=16,
      average-pooling factors 4 and 8, and dropout=0.5.

    The depthwise spatial kernel and classifier use the max-norm constraints
    from the original EEGNet formulation. Call ``apply_max_norm`` after each
    optimizer update.
    """

    def __init__(
        self,
        n_channels: int = 60,
        n_samples: int = 256,
        n_classes: int = 10,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 64,
        separable_kernel: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if f2 != f1 * depth_multiplier:
            raise ValueError("EEGNet-8,2 expects F2 == F1 * D")
        if n_samples % 32 != 0:
            raise ValueError("n_samples must be divisible by 4*8")

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.n_classes = int(n_classes)
        self.f1 = int(f1)
        self.depth_multiplier = int(depth_multiplier)
        self.f2 = int(f2)
        self.temporal_kernel = int(temporal_kernel)
        self.separable_kernel = int(separable_kernel)

        self.temporal_conv = nn.Conv2d(
            1, f1, kernel_size=(1, temporal_kernel), bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1)

        self.spatial_depthwise = nn.Conv2d(
            f1,
            f1 * depth_multiplier,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.separable_depthwise = nn.Conv2d(
            f1 * depth_multiplier,
            f1 * depth_multiplier,
            kernel_size=(1, separable_kernel),
            groups=f1 * depth_multiplier,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(
            f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout)

        feature_dim = f2 * (n_samples // 32)
        self.classifier = nn.Linear(feature_dim, n_classes)

        self.reset_parameters()
        self.apply_max_norm()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        # Per-output-filter L2 constraints, corresponding to the constraints
        # used for the EEGNet depthwise spatial layer and dense classifier.
        spatial = self.spatial_depthwise.weight
        spatial_norm = spatial.norm(p=2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        spatial.mul_(torch.clamp(1.0 / spatial_norm, max=1.0))

        dense = self.classifier.weight
        dense_norm = dense.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        dense.mul_(torch.clamp(0.25 / dense_norm, max=1.0))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG (B,C,T), got {tuple(eeg.shape)}")
        if tuple(eeg.shape[1:]) != (self.n_channels, self.n_samples):
            raise ValueError(
                f"Expected EEG (*,{self.n_channels},{self.n_samples}), "
                f"got {tuple(eeg.shape)}"
            )

        # Match the proposed Stage-3 input normalization: independently
        # z-score each trial and EEG channel over time.
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (
            eeg.std(dim=-1, keepdim=True) + 1e-8
        )
        x = eeg.unsqueeze(1)

        x = _same_pad_1d(x, self.temporal_kernel)
        x = self.bn1(self.temporal_conv(x))
        x = self.spatial_depthwise(x)
        x = self.dropout1(self.pool1(self.elu1(self.bn2(x))))

        x = _same_pad_1d(x, self.separable_kernel)
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.dropout2(self.pool2(self.elu2(self.bn3(x))))

        return self.classifier(torch.flatten(x, start_dim=1))


def architecture_dict(model: EEGNet82) -> dict:
    return {
        "name": "EEGNet-8,2",
        "reference": "Lawhern et al., Journal of Neural Engineering, 2018",
        "doi": "10.1088/1741-2552/aace8c",
        "input_shape": [model.n_channels, model.n_samples],
        "n_classes": model.n_classes,
        "F1": model.f1,
        "D": model.depth_multiplier,
        "F2": model.f2,
        "temporal_kernel": model.temporal_kernel,
        "separable_kernel": model.separable_kernel,
        "pooling": [4, 8],
        "dropout": model.dropout1.p,
        "spatial_max_norm": 1.0,
        "classifier_max_norm": 0.25,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
