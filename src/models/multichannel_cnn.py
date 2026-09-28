"""Stage-1 candidate architecture: Multi-channel 1D CNN.

Input:  (batch, 16, L) -- the 16 Mueller elements kept as separate channels.
Convolution runs only along the wavelength axis, but each filter reads all
16 input channels at once, so the model can learn both spectral (wavelength)
features and cross-channel (Mueller element) relationships without assuming
any spatial adjacency between the 16 elements (unlike a 2D/3D CNN would).
"""
import torch
import torch.nn as nn


class MultiChannelCNN1D(nn.Module):
    def __init__(self, n_channels: int = 16, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)
