import torch.nn as nn
import torch
from argparse import Namespace    
import keras_core as keras
from keras import ops

def get_pruning_layer(config, layer, out_size):
        if config.pruning_method == "dst":
            return DST(config, layer, out_size)
        elif config.pruning_method == "dstkeras":
            return DSTKeras(config, layer, out_size)
        elif config.pruning_method == "autosparse":
            return AutoSparse(config, layer, out_size)
        elif config.pruning_method == "str":
            return STR(config, layer, out_size)
        
def get_threshold_size(config, size, weight_shape):
    if config.threshold_type == "layerwise":
        return (1,1)
    elif config.threshold_type == "channelwise":
        return (size, 1)
    elif config.threshold_type == "weightwise":
        return (weight_shape[0], np.prod(weight_shape[1:]))




class PruningLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(PruningLayer, self).__init__(*args, **kwargs)

    def forward(self, weight):
        pass
    def get_mask(self, weight):
        pass
    def get_layer_sparsity(self, weight):
        with torch.no_grad():
            masked_weight = self.get_mask(weight)
            masked_count = torch.count_nonzero(masked_weight)
            return masked_count / masked_weight.numel()
    def calculate_additional_loss(self, *args, **kwargs):
        return 0
    def post_epoch_function(self, epoch):
        pass


class STR(PruningLayer):
    def __init__(self, config, layer, out_size):
        super(STR, self).__init__()
        self.config = config
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.s = nn.Parameter(torch.ones(threshold_size) * -self.config.threshold_init)
        self.g = torch.sigmoid

    def forward(self, weight):
        mask = self.get_mask(weight)
        return torch.sign(weight) * mask.view(weight.shape) 

    def get_mask(self, weight):
        return torch.relu(torch.abs(weight).view(weight.shape[0], -1) - self.g(self.s))


class AutoSparsePruner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x, threshold, alpha = input
        ctx.save_for_backward((x, alpha))
        mask = torch.relu(torch.abs(x.view(x.shape[0], -1)) - threshold)
        return mask.view(x.shape)
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grads = torch.ones(x.shape)
        grads[x <= 0] = alpha
        return grads * grad_output


class AutoSparse(PruningLayer):
    def __init__(self, config, layer, out_size):
        super(AutoSparse, self).__init__()
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = nn.Parameter(torch.ones(threshold_size) * self.threshold.init)
        self.prune = AutoSparsePruner.apply
        self.config = config

    def forward(self, weight):
        mask = self.get_mask(weight)
        return torch.sign(weight) * mask 

    def get_mask(self, weight):
        return self.prune((weight, self.threshold, self.config.alpha))


class BinaryStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        abs_input = torch.abs(input)
        grads = 2 - 4 * abs_input  
        grads[torch.bitwise_and(abs_input > 0.4, abs_input <= 1.0)] = 0.4
        grads[abs_input > 1.0] = 0.0
        return grad_input * grads


class DST(PruningLayer):
    def __init__(self, config, layer, out_size):
        super(DST, self).__init__()
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = nn.Parameter(torch.zeros(threshold_size))
        self.step = BinaryStep.apply
        self.config = config

    def forward(self, weight):
        mask = self.get_mask(weight)
        ratio = 1 - torch.sum(mask) / mask.numel() # % of pruned weights in tensor
        if ratio >= self.config.max_pruning_pct: # Reset threshold if pruning exceeds ratio
            with torch.no_grad():
                self.threshold.data.fill_(0.)
            mask = self.get_mask(weight)
        masked_weight = weight * mask 
        return masked_weight
    
    def get_mask(self, weight):
        weights_orig_shape = weight.shape
        weights_reshaped = weight.view(weights_orig_shape[0], -1)
        pre_binarystep_weights = torch.abs(weights_reshaped) - self.threshold
        mask = self.step(pre_binarystep_weights)
        mask = mask.view(weights_orig_shape)
        return mask
    
    def calculate_additional_loss(self):
        return self.config.alpha * torch.sum(torch.exp(-self.threshold))


 ####### KERAS: THESE DO NOT WORK #######
@ops.custom_gradient
def binary_step(weight):
    output = ops.cast(weight > 0, dtype=weight.dtype)
    def grad(*args, upstream=None):
        abs_weight = ops.abs(weight)
        idx_lt04 = ops.where(abs_weight <= 0.4, 2 - 4 * abs_weight, 0.0)
        idx_04to1 = ops.where(ops.logical_and(abs_weight > 0.4, abs_weight <=1.0), 0.4, 0.0)
        idx_gt1 = ops.where(abs_weight > 1.0, 0.0, 0.0)
        grads = idx_lt04 + idx_04to1 + idx_gt1
        return grads * upstream
    return output, grad


class DSTKeras(keras.layers.Layer):

    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(DSTKeras, self).__init__(*args, **kwargs)
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = self.add_weight(shape = threshold_size, initializer="zeros", trainable=True) # This does not learn for some reason. Stays 0, even if implementation is the same as with PyTorch
        self.config = config

    def call(self, weight):
        mask = self.get_mask(weight)
        ratio = 1.0 - ops.sum(mask) / ops.size(mask)
        if ratio >= self.config.max_pruning_pct:
            self.threshold.assign(ops.zeros(self.threshold.shape))
            mask = self.get_mask(weight)
        masked_weight = weight * mask
        return masked_weight
    
    def get_mask(self, weight):
        weight_orig_shape = weight.shape
        weights_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        pre_binarystep_weights = ops.abs(weights_reshaped) - self.threshold
        mask = binary_step(pre_binarystep_weights)
        mask = ops.reshape(mask, weight_orig_shape)
        return mask

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_mask(weight)) / ops.size(weight)
    
    def calculate_additional_loss(self):
        return self.config.alpha * ops.sum(ops.exp(-self.threshold))

    def post_epoch_function(self, epoch):
        pass


#### TESTS ####

def test_autosparse_gradient():
    from pruning_methods import AutoSparsePruner

    single_autopruner = AutoSparsePruner
    weight_tensor = torch.Tensor([-2, -1, -0.5, 0, 0.5, 1, 2])
    alpha = torch.Tensor([0.1])

    # 1 so we get only the gradients of the backward itself
    grad_output = torch.ones(1,)
    ctx = {"saved_tensors": (weight_tensor, alpha)}
    ctx = Namespace(**ctx)
    grads = single_autopruner.backward(ctx, grad_output)
    
    # Weights above 0 should have a gradient of 1
    above_zero = (weight_tensor > 0).long()
    gradients_one = (grads == 1).long()
    assert torch.equal(above_zero, gradients_one)

    # Weights at or below 0 should have a gradient equal to alpha
    lte_zero = (weight_tensor <= 0).long()
    gradients_alpha = (grads == alpha).long()
    assert torch.equal(lte_zero, gradients_alpha)

