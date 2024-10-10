import torch
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
from keras import ops
import numpy as np
from utils import cosine_sigmoid_decay


def get_pruning_layer(config, layer, out_size):
        if config.pruning_method == "dst":
            return DST(config, layer, out_size)
        elif config.pruning_method == "autosparse":
            return AutoSparse(config, layer, out_size)
        elif config.pruning_method == "cs":
            return ContinuousSparsification(config, layer, out_size)


def get_threshold_size(config, size, weight_shape):
    if config.threshold_type == "layerwise":
        return (1,1)
    elif config.threshold_type == "channelwise":
        return (size, 1)
    elif config.threshold_type == "weightwise":
        return (weight_shape[0], np.prod(weight_shape[1:]))

        
class ContinuousSparsification(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(ContinuousSparsification, self).__init__(*args, **kwargs)
        self.config = config
        self.beta = config.beta
        self.s = self.add_weight(layer.weight.shape, initializer="ones")
        self.s.assign(config.threshold_init * self.s)
        self.s_init = config.threshold_init
        self.final_temp = config.final_temp
        self.init_weight = layer.weight.clone()
        self.do_hard_mask = False

    def call(self, weight):
        mask = self.get_mask(weight)
        self.mask = mask
        return mask * weight

    def pre_finetune_function(self):
        self.do_hard_mask = True
    
    def get_mask(self, weight):
        scaling = 1. / ops.sigmoid(self.config.threshold_init)
        if self.do_hard_mask:
            mask = self.get_hard_mask(weight)
        else:
            mask = ops.sigmoid(self.beta * self.s) 
        return mask * scaling
    
    def post_epoch_function(self, epoch, total_epochs):
        self.beta *= (self.final_temp**(epoch/(total_epochs - 1)))

    def get_hard_mask(self, weight):
        return (weight > 0).float()

    def post_round_function(self):
        min_beta_s_s0 = ops.minimum(self.beta * self.s, self.s_init)
        self.s.assign(min_beta_s_s0)
        self.beta = 1

    def calculate_additional_loss(self):
        return 0.00000001 * ops.norm(ops.reshape(self.get_mask(self.mask), -1), ord=1)
    
    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask(weight)) / ops.size(weight)


@ops.custom_gradient
def autosparse_prune(x, alpha):
    mask = ops.relu(x)
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        grads = ops.where(x <= 0, alpha, 1.0)
        return grads * upstream, None
    return mask, grad

class AutoSparse(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(AutoSparse, self).__init__(*args, **kwargs)
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = self.add_weight(name="threshold", shape=threshold_size, initializer="ones", trainable=True)
        self.threshold.assign(config.threshold_init * self.threshold)
        self.alpha = ops.convert_to_tensor(config.alpha, dtype="float32")
        self.g = ops.sigmoid

    def call(self, weight):
        """
        sign(W) * ReLu(X), where X = |W| - sigmoid(threshold), with gradient:
            1 if W > 0 else alpha. Alpha is decayed after each epoch.
        """
        mask = self.get_mask(weight)
        return ops.sign(weight) * ops.reshape(mask, weight.shape)

    def get_mask(self, weight):
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1)) 
        w_t = ops.abs(weight_reshaped) - self.g(self.threshold)
        return autosparse_prune(w_t, self.alpha)

    def get_layer_sparsity(self, weight):
        masked_weight = self.get_mask(weight)
        masked_count = ops.count_nonzero(masked_weight)
        return masked_count / ops.size(weight)
    
    def calculate_additional_loss(*args, **kwargs):
        return 0
    
    def post_epoch_function(self, epoch, total_epochs, *args, **kwargs):
        # Decay alpha
        self.alpha = self.alpha * cosine_sigmoid_decay(epoch, total_epochs)


@ops.custom_gradient
def binary_step(weight):
    output = ops.cast(weight > 0, dtype=weight.dtype)
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        abs_weight = ops.abs(weight)
        idx_lt04 = ops.where(abs_weight <= 0.4, 2 - 4 * abs_weight, 0.0)
        idx_04to1 = ops.where(ops.logical_and(abs_weight > 0.4, abs_weight <=1.0), 0.4, 0.0)
        idx_gt1 = ops.where(abs_weight > 1.0, 0.0, 0.0)
        grads = idx_lt04 + idx_04to1 + idx_gt1
        return grads * upstream
    return output, grad

class DST(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(DST, self).__init__(*args, **kwargs)
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = self.add_weight(shape = threshold_size, initializer="zeros", trainable=True)
        self.config = config

    def call(self, weight):
        """
        ReLu(|W| - T), with gradient:
            2 - 4*|W| if |W| <= 0.4
            0.4           if 0.4 < |W| <= 1
            0             if |W| > 1
        """
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

    def post_epoch_function(self, epoch, total_epochs):
        pass


def test_autosparse_gradient():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = torch.tensor(0.4, requires_grad=False).to(device)
    weights = [-2., -1., -0.5, 0., 0.5, 1., 2.]
    expected_output = [0., 0., 0., 0., 0.5, 1., 2.]
    expected_gradient = [alpha, alpha, alpha, alpha, 1., 1., 1.]
    for i, w in enumerate(weights):
        t = torch.tensor([w], requires_grad=True).to(device)
        forward_output, = autosparse_prune(t, alpha)
        backward_output, = torch.autograd.grad(forward_output, t)

        assert forward_output == expected_output[i]
        assert backward_output == expected_gradient[i]
    print("AutoSparse gradient tests passed")


def test_binarystep_gradient():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = [-2., -1., -0.5, 0.25, 0.7, 1., 2.]
    expected_output = [0., 0., 0., 1., 1., 1., 1.]
    expected_gradient = [0., 0.4, 0.4, 1., 0.4, 0.4, 0.]
    for i, w in enumerate(weights):
        t = torch.tensor([w], requires_grad=True).to(device)
        forward_output, = binary_step(t)
        backward_output, = torch.autograd.grad(forward_output, t)
        assert forward_output == expected_output[i]
        assert backward_output == expected_gradient[i]
    print("BinaryStep gradient tests passed")


if __name__ == "__main__":
    test_autosparse_gradient()
    test_binarystep_gradient()