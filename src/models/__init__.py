from .cnn_nd import SpectrumCNN2D, SpectrumCNN3D
from .derivative import WithDerivatives
from .mlp import SpectrumMLP
from .multichannel_cnn import MultiChannelCNN1D

__all__ = ["MultiChannelCNN1D", "SpectrumCNN2D", "SpectrumCNN3D", "SpectrumMLP", "WithDerivatives", "build_model"]

MODELS = {"cnn": MultiChannelCNN1D, "mlp": SpectrumMLP, "cnn2d": SpectrumCNN2D, "cnn3d": SpectrumCNN3D}


def build_model(arch: dict):
    """Model from the `arch` dict train.py saves into a checkpoint. `model`
    picks the class ("cnn" when absent, for checkpoints from before the MLP
    baseline existed); `derivative` (0 when absent) wraps it in
    WithDerivatives; the rest are the class's constructor kwargs."""
    kwargs = dict(arch)
    kind = kwargs.pop("model", "cnn")
    order = kwargs.pop("derivative", 0)
    if kind not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {kind!r}")
    if order == 0:
        return MODELS[kind](**kwargs)
    if kind not in ("cnn", "mlp"):
        raise ValueError(f"derivative features are only wired up for 'cnn' and 'mlp', got {kind!r}")
    n_channels = kwargs["n_channels"]
    kwargs["n_channels"] = n_channels * (1 + order)
    return WithDerivatives(MODELS[kind](**kwargs), n_channels, order)
