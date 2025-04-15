import torch
import torch.nn as nn


class ActivationPruning(nn.Module):

    def __init__(self, config, layer, out_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.act_type = "relu"
        self.t = 0
        self.layer_type = "linear" if isinstance(layer, nn.Linear) else "conv"
        self.shape = (layer.weight.shape[0], 1)
        if self.layer_type == "conv":
            self.shape = (layer.weight.shape[0], 1, 1, 1)
        self.mask = torch.ones(self.shape, requires_grad=False).to(layer.weight.device)
        self.activations = None
        self.total = 0.0
        self.is_pretraining = True

    def collect_output(self, output):
        """
        Accumulates values for how often the outputs of the neurons and channels of
        linear/convolution layer are over 0. Every t_delta steps, uses these values to update
        the mask to prune those channels and neurons that are active less than a given threshold
        """
        if not self.training or self.is_pretraining:
            # Don't collect during validation
            return
        if self.activations is None:
            # Initialize activations dynamically
            self.activations = torch.zeros(size=output.shape[1:], dtype=output.dtype, device=self.mask.device)
        self.t += 1
        self.total += output.shape[0]
        gt_zero = (output > 0).float()
        gt_zero = torch.sum(gt_zero, dim=0)  # Sum over batch, take average during mask update
        self.activations += gt_zero
        if self.t % self.config["pruning_parameters"]["t_delta"] == 0:
            pct_active = self.activations / self.total
            self.t = 0
            self.total = 0
            if self.layer_type == "linear":
                self.mask = (pct_active > self.config["pruning_parameters"]["threshold"]).float().unsqueeze(1)
            else:
                pct_active = pct_active.view(pct_active.shape[0], -1)
                pct_active_avg = torch.mean(pct_active, dim=-1)
                pct_active_above_threshold = (pct_active_avg > self.config["pruning_parameters"]["threshold"]).float()
                self.mask = (pct_active_above_threshold).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.activations *= 0.0

    def build(self, weight):
        # Since this is a torch layer, do nothing
        pass

    def forward(self, weight):  # Mask is only updated every t_delta step, using collect_output
        return self.mask * weight

    def post_pre_train_function(self):
        self.is_pretraining = False

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def post_round_function(self):
        pass

    def pre_finetune_function(self):
        pass

    def calculate_additional_loss(self):
        return 0

    def get_layer_sparsity(self, weight):
        pass

    def post_epoch_function(self, epoch, total_epochs):
        pass
