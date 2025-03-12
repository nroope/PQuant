import yaml

from pquant.pruning_methods.dst import DST
from pquant.pruning_methods.autosparse import AutoSparse
from pquant.pruning_methods.cs import ContinuousSparsification
from pquant.pruning_methods.pdp import PDP
from pquant.pruning_methods.activation_pruning import ActivationPruning
from pquant.pruning_methods.wanda import Wanda


def get_pruning_layer(config, layer, out_size):
        if config.pruning_method == "dst":
            return DST(config, layer, out_size)
        elif config.pruning_method == "autosparse":
            return AutoSparse(config, layer, out_size)
        elif config.pruning_method == "cs":
            return ContinuousSparsification(config, layer, out_size)
        elif config.pruning_method == "pdp":
            return PDP(config, layer, out_size)
        elif config.pruning_method == "continual_learning":
            return ActivationPruning(config, layer, out_size)
        elif config.pruning_method == "wanda":
            return Wanda(config, layer, out_size)


def get_pruning_config(config_path):    
    with open(config_path, "r") as f:
        pruning_config = yaml.safe_load(f)
        params = pruning_config["pruning_parameters"] | pruning_config["training_parameters"]
        params = params | pruning_config["quantization_parameters"]
        return params