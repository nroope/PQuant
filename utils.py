from keras_core import ops
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

pi = ops.convert_to_tensor(np.pi)
L0 = ops.convert_to_tensor(-6.0)
L1 = ops.convert_to_tensor(6.0)

def cosine_decay(i, T):
    return (1 + ops.cos(pi * i / T)) / 2

def sigmoid_decay(i, T):
    return 1 - ops.sigmoid(L0 + (L1 - L0) * i / T)

def cosine_sigmoid_decay(i, T):
    return ops.maximum(cosine_decay(i, T), sigmoid_decay(i, T))

############## OPTIMIZERS AND SCHEDULERS ##############
def get_optimizer(config, model):
    if config.optimizer == "sgd":
        parameters = list(model.named_parameters())
        threshold_params = [v for n, v in parameters if "threshold" in n and v.requires_grad]
        rest_params = [v for n, v in parameters if "threshold" not in n and v.requires_grad]
        optimizer = torch.optim.SGD(
            [{
                "params": threshold_params,
                "weight_decay": config.threshold_decay if config.threshold_decay is not None else config.l2_decay,
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
