import torch
import torch.nn as nn

class Wanda(nn.Module):

    def __init__(self, config, layer, out_size, *args, **kwargs):
            super(Wanda, self).__init__(*args, **kwargs)
            self.config = config
            self.act_type = "relu"
            self.t = 0
            self.layer_type = "linear" if isinstance(layer, nn.Linear) else "conv"
            self.mask = torch.tensor(1.0)
            self.inputs = None
            self.total = 0.
            self.weight = layer.weight
            self.done = False
            self.sparsity = self.config["pruning_parameters"]["sparsity"]
            self.is_pretraining = True
            self.N = self.config["pruning_parameters"]["N"]
            self.M = self.config["pruning_parameters"]["M"]
            self.t_start_collecting = self.config["pruning_parameters"]["t_start_collecting"]


    def handle_linear(self, x):
        norm = x.norm(p=2, dim=0)
        metric = self.weight.abs() * norm
        if self.N is not None and self.M is not None:
            # N:M pruning
            W_mask = torch.zeros_like(self.weight)
            for ii in range(self.weight.shape[1]):
                if ii % self.M == 0:
                    tmp = metric.abs()[:,ii:(ii+self.M)].float()
                    if tmp.shape[1] < self.M:
                        continue
                    indices = ii+torch.topk(tmp, self.N, dim=1, largest=False)[1]
                    W_mask = torch.scatter(W_mask, 1, indices, 1)
                    self.mask.data = W_mask.view(self.weight.shape)
        else:
            # Unstructured pruning
            _, sorted_idx = torch.sort(metric, dim=1)
            pruned_idx = sorted_idx[:,:int(self.weight.shape[1] * self.sparsity)]
            self.weight.data = self.weight.data.scatter_(dim=1, index=pruned_idx, src=torch.zeros(pruned_idx.shape).to(self.weight.device))
            self.mask.data = (self.weight != 0).float()

    def handle_conv(self, x):
        inputs_avg = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=0)
        norm = inputs_avg.norm(p=2, dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        metric = self.weight.abs() * norm
        if self.N is not None and self.M is not None:
            # N:M pruning
            metric = x.view(metric.shape[0], -1)
            W_mask = torch.zeros(self.weight.shape, device=metric.device)
            W_mask = W_mask.view(W_mask.shape[0], -1)
            for ii in range(W_mask.shape[1]):
                if ii % self.M == 0:
                    tmp = metric.abs()[:,ii:(ii+self.M)].float()
                    if tmp.shape[1] < self.M:
                        continue
                    n = min(self.N, W_mask.shape[1]-ii)
                    indices = ii+torch.topk(tmp, n, dim=1, largest=False)[1]
                    indices_limited = torch.minimum(indices, torch.tensor(W_mask.shape[1]-1))
                    W_mask = torch.scatter(W_mask, 1, indices_limited , 1)
            self.mask.data = W_mask.view(self.weight.shape)
        else:
            # Unstructured pruning
            _, sorted_idx = torch.sort(metric, dim=1)
            pruned_idx = sorted_idx[:, :int(self.weight.shape[1] * self.sparsity)]
            self.weight.data = self.weight.data.scatter_(dim=1, index=pruned_idx, src=torch.zeros(pruned_idx.shape).to(self.weight.device))
            self.mask.data = (self.weight != 0).float()

    def collect_input(self, x):
            if self.done or self.is_pretraining:
                return
            """
            Accumulates layer inputs starting at step t_start_collecting for t_delta steps, then averages it. 
            Calculates a metric based on weight absolute values and norm of inputs.
            For linear layers, calculate norm over batch dimension.
            For conv layers, take average over batch dimension and calculate norm over flattened kernel_size dimension.
            If N and M are defined, do N:M pruning.
            """
            if not self.training or x.shape[0] != self.config["training_parameters"]["batch_size"]:
                # Don't collect during validation
                return
            self.t += 1
            if self.t < self.t_start_collecting:
                return
            self.total += x.shape[0]
            self.inputs = x if self.inputs is None else self.inputs + x

            if self.t % (self.t_start_collecting + self.config["pruning_parameters"]["t_delta"]) == 0:
                inputs_avg = self.inputs / self.total
                if self.layer_type == "linear":
                    self.handle_linear(inputs_avg)
                else:
                    self.handle_conv(inputs_avg)
                self.done = True
                self.inputs = None

    def build(self, weight):
        # Since this is a torch layer, do nothing
        pass

    def forward(self, weight): # Mask is only updated every t_delta step, using collect_output
        self.weight.data = weight
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

