"""NN baseline: plain fully-connected network on the flattened spectrum.

No convolution, so no built-in notion of neighbouring wavelengths or of
which values belong to the same channel -- the reference point the
architecture comparison measures the CNN variants against.
"""
import torch
import torch.nn as nn


class SpectrumMLP(nn.Module):
    def __init__(self, n_channels: int = 16, hidden: int = 64, dropout: float = 0.0,
                 n_wavelength: int = 101, n_outputs: int = 1):
        super().__init__()
        width = hidden * 4
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * n_wavelength, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, n_outputs),
        )
        self.n_outputs = n_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.squeeze(-1) if self.n_outputs == 1 else x
