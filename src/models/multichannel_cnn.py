"""Stage-1 candidate architecture: Multi-channel 1D CNN.

Input:  (batch, n_channels, L) -- the 16 Mueller elements (+ TE, TM) kept as
separate channels. Convolution runs only along the wavelength axis, but each
filter reads all input channels at once, so the model can learn both spectral
(wavelength) features and cross-channel relationships without assuming any
spatial adjacency between the Mueller elements (unlike a 2D/3D CNN would).

pooling:
  * "flatten" (default) -- keep the pooled feature map's wavelength axis and
    flatten it into the head. CD is encoded mostly in *where* spectral
    features sit (resonance positions shift with CD), so the head needs to
    know which wavelength a feature came from.
  * "avg" -- legacy: AdaptiveAvgPool1d(1) averages over wavelength, which
    throws that position information away. Kept only so checkpoints trained
    before this change still load (see inference.load_checkpoint).

dropout defaults to 0 -- for this low-noise regression task, dropout right
before the output mostly injects noise into the gradient (legacy training
showed train MSE ~17x val MSE, i.e. dominated by dropout noise).
"""
import torch
import torch.nn as nn


class MultiChannelCNN1D(nn.Module):
    def __init__(self, n_channels: int = 16, hidden: int = 64, dropout: float = 0.0,
                 pooling: str = "flatten", n_wavelength: int = 101):
        super().__init__()
        if pooling not in ("flatten", "avg"):
            raise ValueError(f"pooling must be 'flatten' or 'avg', got {pooling!r}")
        self.pooling = pooling

        layers = [
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
        ]
        if pooling == "avg":
            layers.append(nn.AdaptiveAvgPool1d(1))
            head_in = hidden * 2
        else:
            # Two MaxPool1d(2) layers: L -> L//2 -> L//4.
            head_in = hidden * 2 * ((n_wavelength // 2) // 2)
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)
