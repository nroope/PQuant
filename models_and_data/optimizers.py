import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from pquant.core.p_optim import pAdam, pSGD


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
    total_w = []
    nonzeros = []
    for n, m in sparse_model.named_modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.hist(m.weight.detach().cpu().numpy().flatten())
            names.append(n)
            nonzero = np.count_nonzero(m.weight.detach().cpu())
            remaining_pct = nonzero/ m.weight.numel()
            remaining.append(remaining_pct)
            total_w.append(m.weight.numel())
            nonzeros.append(nonzero)
            ax.title.set_text(f"{n} weight distribution")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{n.replace(".", "_")}_weight_hist.png")
            plt.cla()
            plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(len(names)), remaining)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.tick_params(axis='x', labelrotation=270)
    new_ytick = []
    for i in ax.get_yticklabels():
        ytick = f"{float(i.get_text()) * 100}%"
        new_ytick.append(ytick)
    ax.set_yticklabels(new_ytick)
    ax.title.set_text("Remaining weights per layer")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/remaining_weights_pct.png")
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(len(nonzeros)), total_w, color="lightcoral", label="total weights")
    ax.bar(range(len(nonzeros)), nonzeros, color="steelblue", label="nonzero weights")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.tick_params(axis='x', labelrotation=270)
    ax.title.set_text("Weights per layer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/remaining_weights_total.png")
    plt.cla()
    plt.clf()