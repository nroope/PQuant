from keras_core import ops
import numpy as np
import torch
import yaml
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from argparse import Namespace
from p_optim import pAdam, pSGD

pi = ops.convert_to_tensor(np.pi)
L0 = ops.convert_to_tensor(-6.0)
L1 = ops.convert_to_tensor(6.0)

def cosine_decay(i, T):
    return (1 + ops.cos(pi * i / T)) / 2

def sigmoid_decay(i, T):
    return 1 - ops.sigmoid(L0 + (L1 - L0) * i / T)

def cosine_sigmoid_decay(i, T):
    return ops.maximum(cosine_decay(i, T), sigmoid_decay(i, T))

def get_config_from_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        config_flat = {**config["pruning_parameters"], **config["training_parameters"], **config["not_used"]}
        return Namespace(**config_flat)

def save_config_to_yaml(config, path):
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config.__dict__, f)

############## OPTIMIZERS AND SCHEDULERS ##############
def get_optimizer(config, model):
    if config.optimizer == "sgd":
        # CS already has L1-regularization for threshold parameters
        threshold_decay = 0 if config.pruning_method == "cs" else config.threshold_decay 
    
        parameters = list(model.named_parameters())
        threshold_params = [v for n, v in parameters if "threshold" in n and v.requires_grad]
        rest_params = [v for n, v in parameters if "threshold" not in n and v.requires_grad]
        optimizer = torch.optim.SGD(
            [{
                "params": threshold_params,
                "weight_decay": threshold_decay if threshold_decay is not None else config.l2_decay,
            },
            {   "params": rest_params, 
                 "weight_decay": config.l2_decay
            },
            ],
            config.lr,
            momentum=config.momentum)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "psgd":
        # CS already has L1-regularization for threshold parameters
        threshold_decay = 0 if config.pruning_method == "cs" else config.threshold_decay 
    
        parameters = list(model.named_parameters())
        threshold_params = [v for n, v in parameters if "torch_params" in n and v.requires_grad]
        rest_params = [v for n, v in parameters if "torch_params" not in n and v.requires_grad]
        optimizer = pSGD(
            [{
                "params": threshold_params,
            },
            {   "params": rest_params, 
            },
            ],
            config.lr,
            momentum=config.momentum,
            lambda_p=config.lambda_p,
            p_norm=config.pnorm,
            weight_decay=config.l2_decay)
    elif config.optimizer == "padam":
        optimizer = pAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    return optimizer

def get_scheduler(optimizer, config):
    if config.lr_schedule is None:
        return None
    elif config.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, config.cosine_tmax)
    elif config.lr_schedule == "multistep":
        return MultiStepLR(optimizer, config.milestones, gamma=config.gamma)
    return None
#######################################################

def plot_weights_per_layer(sparse_model, output_dir):
    names = []
    remaining = []
    for n, m in sparse_model.named_modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            names.append(n)
            rem = np.count_nonzero(m.weight.detach().cpu()) / m.weight.numel()
            remaining.append(rem)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(len(names)), remaining)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.tick_params(axis='x', labelrotation=270)
    ax.title.set_text("Remaining weights per layer")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/remaining_weights.png")
    plt.cla()
    plt.clf()