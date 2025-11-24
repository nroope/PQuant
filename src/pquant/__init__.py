import importlib
import os
import sys

# flake8: noqa
backend = os.getenv("KERAS_BACKEND", "tensorflow")
if backend == "torch":
    from . import configs, pruning_methods
    from .core.torch import activations, layers, optimizers, quantizer
    from .core.torch.layers import add_compression_layers, post_training_prune
    from .core.torch.train import train_model

    _forwards = ["activations", "layers", "quantizer", "optimizers"]

    for name in _forwards:
        mod = importlib.import_module(f".core.torch.{name}", package="pquant")
        sys.modules[f"{__name__}.{name}"] = mod
        setattr(sys.modules[__name__], name, mod)

    _forwards.append("train_model")
    _forwards.append("add_compression_layers")
    _forwards.append("configs")
    _forwards.append("pruning_methods")
    _forwards.append("post_training_prune")
    __all__ = _forwards

else:
    from . import configs, pruning_methods
    from .core.keras import activations, layers, quantizer
    from .core.keras.layers import add_compression_layers, post_training_prune
    from .core.keras.train import train_model

    _forwards = ["activations", "layers", "quantizer"]

    for name in _forwards:
        mod = importlib.import_module(f".core.keras.{name}", package="pquant")
        sys.modules[f"{__name__}.{name}"] = mod
        setattr(sys.modules[__name__], name, mod)

    _forwards.append("train_model")
    _forwards.append("add_compression_layers")
    _forwards.append("configs")
    _forwards.append("pruning_methods")
    _forwards.append("post_training_prune")
    __all__ = _forwards
