from keras_core import ops
import numpy as np
import torch
import yaml
import os
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
            momentum=config.momentum,
            weight_decay=config.l2_decay)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    elif config.optimizer == "psgd":
        # CS already has L1-regularization for threshold parameters
        threshold_decay = 0 if config.pruning_method == "cs" else config.threshold_decay 
    
        parameters = list(model.named_parameters())
        threshold_params = [v for n, v in parameters if "threshold" in n and v.requires_grad]
        rest_params = [v for n, v in parameters if "threshold" not in n and v.requires_grad]
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
