import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import ops
import numpy as np

pi = ops.convert_to_tensor(np.pi)
L0 = ops.convert_to_tensor(-6.0)
L1 = ops.convert_to_tensor(6.0)

def cosine_decay(i, T):
    return (1 + ops.cos(pi * i / T)) / 2

def sigmoid_decay(i, T):
    return 1 - ops.sigmoid(L0 + (L1 - L0) * i / T)

def cosine_sigmoid_decay(i, T):
    return ops.maximum(cosine_decay(i, T), sigmoid_decay(i, T))

def get_threshold_size(config, size, weight_shape):
    if config["pruning_parameters"]["threshold_type"] == "layerwise":
        return (1,1)
    elif config["pruning_parameters"]["threshold_type"] == "channelwise":
        return (size, 1)
    elif config["pruning_parameters"]["threshold_type"] == "weightwise":
        return (weight_shape[0], np.prod(weight_shape[1:]))

BACKWARD_SPARSITY = False

@ops.custom_gradient
def autosparse_prune(x, alpha):
    mask = ops.relu(x)
    backward_sparsity = 0.5
    x_flat = ops.reshape(x, -1)
    k = int(ops.size(x_flat) * backward_sparsity)
    topks, _ = ops.top_k(x_flat, k)
    kth_value = topks[-1]
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        grads = ops.where(x <= 0, alpha, 1.0)
        if BACKWARD_SPARSITY:
            grads = ops.where(x < kth_value, 0., grads)
        return grads * upstream, None
    return mask, grad

class AutoSparse(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(AutoSparse, self).__init__(*args, **kwargs)
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = self.add_weight(name="threshold", shape=threshold_size, initializer="ones", trainable=True)
        self.threshold.assign(config["pruning_parameters"]["threshold_init"] * self.threshold)
        self.alpha = ops.convert_to_tensor(config["pruning_parameters"]["alpha"], dtype="float32")
        self.g = ops.sigmoid
        self.config = config
        global BACKWARD_SPARSITY
        BACKWARD_SPARSITY = config["pruning_parameters"]["backward_sparsity"]

    def call(self, weight):
        """
        sign(W) * ReLu(X), where X = |W| - sigmoid(threshold), with gradient:
            1 if W > 0 else alpha. Alpha is decayed after each epoch.
        """
        mask = self.get_mask(weight)
        self.mask = ops.reshape(mask, weight.shape)
        return ops.sign(weight) * ops.reshape(mask, weight.shape)

    def get_hard_mask(self, weight):
        return self.mask

    def get_mask(self, weight):
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1)) 
        w_t = ops.abs(weight_reshaped) - self.g(self.threshold)
        return autosparse_prune(w_t, self.alpha)

    def get_layer_sparsity(self, weight):
        masked_weight = self.get_mask(weight)
        masked_count = ops.count_nonzero(masked_weight)
        return masked_count / ops.size(weight)
    
    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def calculate_additional_loss(*args, **kwargs):
        return 0

    def pre_finetune_function(self):
        pass

    def post_epoch_function(self, epoch, total_epochs, alpha_multiplier, autotune_epochs=0, writer=None, global_step=0):
        # Decay alpha
        if epoch >= autotune_epochs:
            self.alpha *= cosine_sigmoid_decay(epoch - autotune_epochs, total_epochs)
        else:
            self.alpha *= alpha_multiplier
        if epoch == self.config["pruning_parameters"]["alpha_reset_epoch"]:
            self.alpha *= 0.
        if writer is not None:
            writer.write_scalars([(f"Autosparse_alpha", self.alpha, global_step)])