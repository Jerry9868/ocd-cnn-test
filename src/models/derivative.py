"""Derivative features: feed a model the spectrum plus its wavelength derivatives.

CD mostly moves spectral features sideways, which shows up directly in
d(spectrum)/d(wavelength). WithDerivatives appends the first (order 1) or
first and second (order 2) finite differences of every channel as extra
input channels, so the wrapped model sees n_channels x (1 + order).

Differencing shrinks the signal and amplifies noise by different amounts
per channel, so the derivative channels pass through a BatchNorm without
learnable scale to bring them back to unit variance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WithDerivatives(nn.Module):
    def __init__(self, base: nn.Module, n_channels: int, order: int):
        super().__init__()
        if order not in (1, 2):
            raise ValueError(f"derivative order must be 1 or 2, got {order}")
        self.base = base
        self.order = order
        self.scale = nn.BatchNorm1d(n_channels * order, affine=False)
        self.n_outputs = getattr(base, "n_outputs", 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (1, 1), mode="replicate")
        diffs = [(padded[:, :, 2:] - padded[:, :, :-2]) / 2]
        if self.order == 2:
            diffs.append(padded[:, :, 2:] - 2 * x + padded[:, :, :-2])
        return self.base(torch.cat([x, self.scale(torch.cat(diffs, dim=1))], dim=1))
