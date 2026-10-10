"""2D and 3D CNN candidates for the architecture comparison.

Both add an assumption the Multi-channel 1D CNN deliberately avoids: that
Mueller elements which are neighbours in the matrix are related the way
neighbouring pixels are.

  * SpectrumCNN2D -- the spectrum as one image, channels x wavelength.
    Filters are 3 channels tall, so a filter only sees elements adjacent in
    the channel order (M00, M01, ..., M33, TE, TM).
  * SpectrumCNN3D -- the 16 Mueller elements as a 4 x 4 x wavelength volume.
    Filters are 3 x 3 across the matrix. Any channels beyond the first 16
    (TE, TM) have no place in the matrix, so each is repeated over the 4 x 4
    grid as an extra input feature.

Widths are a fraction of the 1D CNN's for the same `hidden` (2D: 1/3, 3D:
1/6). Convolving over the extra axes costs that much more per filter; at
equal widths a CPU training run takes hours.
"""
import torch
import torch.nn as nn

N_MUELLER = 16


class SpectrumCNN2D(nn.Module):
    def __init__(self, n_channels: int = 16, hidden: int = 64, dropout: float = 0.0,
                 n_wavelength: int = 101, n_outputs: int = 1):
        super().__init__()
        self.n_outputs = n_outputs
        widths = (hidden // 6, hidden // 3, hidden * 2 // 3)
        layers = []
        in_ch = 1
        for i, (out_ch, kernel) in enumerate(zip(widths, (7, 5, 3))):
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=(3, kernel), padding=(1, kernel // 2)),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            if i < 2:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        head_in = widths[2] * ((n_channels // 2) // 2) * ((n_wavelength // 2) // 2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(head_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.features(x[:, None]))
        return x.squeeze(-1) if self.n_outputs == 1 else x


class SpectrumCNN3D(nn.Module):
    def __init__(self, n_channels: int = 16, hidden: int = 64, dropout: float = 0.0,
                 n_wavelength: int = 101, n_outputs: int = 1):
        super().__init__()
        if n_channels < N_MUELLER:
            raise ValueError(f"cnn3d needs the 16 Mueller elements as the first channels, got {n_channels}")
        self.n_outputs = n_outputs
        widths = (hidden // 12, hidden // 6, hidden // 3)
        layers = []
        in_ch = 1 + n_channels - N_MUELLER
        for i, (out_ch, kernel) in enumerate(zip(widths, (7, 5, 3))):
            layers += [nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, kernel), padding=(1, 1, kernel // 2)),
                       nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)]
            # Wavelength is halved twice, as in the 1D CNN; the 4 x 4 matrix once, at the end.
            layers.append(nn.MaxPool3d((1, 1, 2)) if i < 2 else nn.MaxPool3d((2, 2, 1)))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        head_in = widths[2] * 2 * 2 * ((n_wavelength // 2) // 2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(head_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        volume = x[:, :N_MUELLER].reshape(batch, 1, 4, 4, length)
        extra = x[:, N_MUELLER:, None, None, :].expand(-1, -1, 4, 4, -1)
        x = self.head(self.features(torch.cat([volume, extra], dim=1)))
        return x.squeeze(-1) if self.n_outputs == 1 else x
