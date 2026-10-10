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

downsample / norm / conv_groups exist for the architecture ablation
(experiments/ablation.py); the defaults reproduce the original network:
  * downsample: "maxpool" (default) | "avgpool" | "none" -- how the two
    2x reductions along wavelength are done, or skipped entirely.
  * norm: "batch" (default) | "none".
  * conv_groups: "full" (default) -- every filter reads all input channels;
    "per_channel" -- each input channel gets its own filters all the way to
    the head (no cross-channel mixing in the conv stack), with widths rounded
    up to a multiple of n_channels. The head still sees every channel, so
    this isolates whether *convolutional* cross-channel features matter.

attention: "none" (default) | "input" -- Channel Attention on the input: a
small squeeze-and-excitation gate (InputChannelAttention) rescales each of
the n_channels spectra before the first convolution, from a summary of the
whole spectrum. The gate values say how much the network leans on each
Mueller element for a given spectrum (channel_attention()).

dropout defaults to 0 -- for this low-noise regression task, dropout right
before the output mostly injects noise into the gradient (legacy training
showed train MSE ~17x val MSE, i.e. dominated by dropout noise).
"""
import torch
import torch.nn as nn


class InputChannelAttention(nn.Module):
    """(batch, n_channels, L) -> (batch, n_channels) gates in (0, 2), 1 at
    initialization, from each channel's mean and spread over wavelength."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_channels, 2 * n_channels), nn.ReLU(inplace=True),
            nn.Linear(2 * n_channels, n_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return 2 * torch.sigmoid(self.net(summary))


class MultiChannelCNN1D(nn.Module):
    def __init__(self, n_channels: int = 16, hidden: int = 64, dropout: float = 0.0,
                 pooling: str = "flatten", n_wavelength: int = 101,
                 downsample: str = "maxpool", norm: str = "batch", conv_groups: str = "full",
                 n_outputs: int = 1, attention: str = "none"):
        super().__init__()
        if attention not in ("none", "input"):
            raise ValueError(f"attention must be 'none' or 'input', got {attention!r}")
        self.attention = InputChannelAttention(n_channels) if attention == "input" else None
        if pooling not in ("flatten", "avg"):
            raise ValueError(f"pooling must be 'flatten' or 'avg', got {pooling!r}")
        if downsample not in ("maxpool", "avgpool", "none"):
            raise ValueError(f"downsample must be 'maxpool', 'avgpool' or 'none', got {downsample!r}")
        if norm not in ("batch", "none"):
            raise ValueError(f"norm must be 'batch' or 'none', got {norm!r}")
        if conv_groups not in ("full", "per_channel"):
            raise ValueError(f"conv_groups must be 'full' or 'per_channel', got {conv_groups!r}")
        self.pooling = pooling
        self.n_outputs = n_outputs

        groups = n_channels if conv_groups == "per_channel" else 1
        # Grouped convs need every width divisible by the group count.
        widths = [-(-w // groups) * groups for w in (hidden // 2, hidden, hidden * 2)]

        layers = []
        in_ch = n_channels
        for i, (out_ch, kernel) in enumerate(zip(widths, (7, 5, 3))):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, groups=groups))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if i < 2 and downsample != "none":
                layers.append(nn.MaxPool1d(2) if downsample == "maxpool" else nn.AvgPool1d(2))
            in_ch = out_ch

        if pooling == "avg":
            layers.append(nn.AdaptiveAvgPool1d(1))
            head_in = widths[2]
        else:
            # Two 2x reductions: L -> L//2 -> L//4.
            reduced = n_wavelength if downsample == "none" else (n_wavelength // 2) // 2
            head_in = widths[2] * reduced
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    def channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_channels) input gates; all ones without attention."""
        if self.attention is None:
            return torch.ones(x.shape[:2])
        return self.attention(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention is not None:
            x = x * self.attention(x)[:, :, None]
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1) if self.n_outputs == 1 else x
