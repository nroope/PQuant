from argparse import ArgumentParser, Namespace, ArgumentTypeError
import yaml
import os

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
    config = config | get_particle_transformer_model_config()
    config = config | get_pruning_config(config["pruning_config"])
    config = Namespace(**config)

    return config

def write_config_to_yaml(config, output_dir):
    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.dump(config.__dict__, f)

def read_config_yaml(config):
    path = f"{config["validation_config_folder"]}/config.yaml"
    with open(path, "r") as f:
        val_config = yaml.safe_load(f)
        val_config["validation_config_folder"] = config["validation_config_folder"]
        return Namespace(**val_config)

def get_particle_transformer_model_config():
    with open("particle_transformer/particle_transformer.yaml", "r") as f:
        model_config = yaml.safe_load(f)
        print(model_config)
        model_config["local_rank"] = None if model_config["backend"] is None else int(os.environ.get("LOCAL_RANK", "0"))
        return model_config

def get_config_from_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        config_flat = {**config["pruning_parameters"], **config["training_parameters"], **config["not_used"]}
        return Namespace(**config_flat)

def save_config_to_yaml(config, path):
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config.__dict__, f)

    return {}