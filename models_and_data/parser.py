from argparse import ArgumentParser, ArgumentTypeError
import yaml

from pquant.core.utils import get_pruning_config

def str2bool(w):
    if w.lower() in ['true', 'y', 'yes']:
        return True
    elif w.lower() in ['false', 'no', 'n']:
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def parse_cmdline_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--dataset", default=None, help="If not using data from config, use this dataset")
    parser.add_argument("--model", default="resnet18", help="If not using model from config, use this model.")
    parser.add_argument("--pruning_config", type=str, default=None, help="Path to pruning config file")
    parser.add_argument("--validation_config_folder", type=str, default=None)
    config = parser.parse_args(args=args).__dict__
    if config["validation_config_folder"] is not None: # Use validation config instead
        return read_config_yaml(config)
    config = config | get_pruning_config(config["pruning_config"])
    return config

def read_config_yaml(config):
    path = f"{config["validation_config_folder"]}/config.yaml"
    with open(path, "r") as f:
        val_config = yaml.safe_load(f)
        val_config["validation_config_folder"] = config["validation_config_folder"]
        return val_config