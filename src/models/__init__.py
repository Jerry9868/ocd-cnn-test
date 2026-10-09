from .mlp import SpectrumMLP
from .multichannel_cnn import MultiChannelCNN1D

__all__ = ["MultiChannelCNN1D", "SpectrumMLP", "build_model"]


def build_model(arch: dict):
    """Model from the `arch` dict train.py saves into a checkpoint. `model`
    picks the class ("cnn" when absent, for checkpoints from before the MLP
    baseline existed); the rest are its constructor kwargs."""
    kwargs = dict(arch)
    kind = kwargs.pop("model", "cnn")
    if kind == "cnn":
        return MultiChannelCNN1D(**kwargs)
    if kind == "mlp":
        return SpectrumMLP(**kwargs)
    raise ValueError(f"model must be 'cnn' or 'mlp', got {kind!r}")
