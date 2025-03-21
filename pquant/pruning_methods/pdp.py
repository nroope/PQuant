import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import ops
import torch.nn as nn

class PDP(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(PDP, self).__init__(*args, **kwargs)
        self.init_r = config["pruning_parameters"]["sparsity"]
        self.r = config["pruning_parameters"]["sparsity"]
        self.temp = config["pruning_parameters"]["temperature"]
        self.is_pretraining = True
        self.config = config
        self.fine_tuning = False
        self.layer_type = "linear" if isinstance(layer, nn.Linear) else "conv"

    def build(self, input_shape):
        input_shape_concatenated = list(input_shape) + [1]
        self.softmax_shape = input_shape_concatenated
        self.t = ops.ones(input_shape_concatenated) * 0.5
        self.mask = ops.ones(input_shape)
        super().build(input_shape)

    def post_pre_train_function(self):
        self.is_pretraining = False # Enables pruning

    def pre_epoch_function(self, epoch, total_epochs):
        if not self.is_pretraining:
            self.r = ops.minimum(1., self.config["pruning_parameters"]["epsilon"] * (epoch + 1)) * self.init_r
    
    def post_round_function(self):
        pass

    def get_hard_mask(self, weight):
        if self.fine_tuning:
            return (self.mask >= 0.5).float()
        if self.config["pruning_parameters"]["structured_pruning"]:
            if self.layer_type == "conv":
                mask = self.get_mask_structured_channel(weight)
            else:
                mask = self.get_mask_structured_linear(weight)
        else:
            mask = self.get_mask(weight)
        self.mask = (mask >= 0.5).float()
        return (mask >= 0.5).float()
    
    def pre_finetune_function(self):
        self.fine_tuning = True
        self.mask = (self.mask  >= 0.5).float()

    def get_mask_structured_linear(self, weight):
        """
        Structured pruning. Use the l2 norm of the neurons instead of the absolute weight values to calculate threshold point t. 
        Prunes whole neurons.
        """
        if self.is_pretraining:
            return self.mask
        norm = ops.norm(weight, axis=0, ord=2, keepdims=True)
        norm_flat = ops.reshape(norm, -1)
        # Do top_k for all neuron norms. Returns sorted array, just use the values on both sides of the threshold (sparsity * size(norm)) to calculate t directly
        W_all, _ = ops.top_k(norm_flat, ops.size(norm_flat))
        lim = ops.clip(int((1-self.r) * ops.size(W_all)), 0, ops.size(W_all) - 2)

        Wh = W_all[lim]
        Wt = W_all[lim + 1]
        #norm = ops.expand_dims(norm, -1)
        t = ops.ones(norm.shape) * 0.5 * (Wh + Wt) 
        soft_input = ops.concatenate((t ** 2, norm ** 2), axis=0) / self.temp
        softmax_result = ops.softmax(soft_input, axis=0)
        zw, mw = ops.unstack(softmax_result, axis=0)
        mw = ops.expand_dims(mw, 0)
        self.mask = mw
        return mw

    def get_mask_structured_channel(self, weight):
        """
        Structured pruning. Use the l2 norm of the channels instead of the absolute weight values to calculate threshold point t.
        Prunes whole channels.
        """
        if self.is_pretraining:
            return self.mask
        weight_reshaped = ops.reshape(weight, (weight.shape[0], weight.shape[1], -1))
        norm = ops.norm(weight_reshaped, axis=2, ord=2)
        norm_flat = ops.reshape(norm, -1)
        # Do top_k for all channel norms. Returns sorted array, just use the values on both sides of the threshold (sparsity * size(norm)) to calculate t directly

        W_all, _ = ops.top_k(norm_flat, ops.size(norm_flat))
        lim = ops.clip(int((1-self.r) * ops.size(W_all)), 0, ops.size(W_all) - 2)

        Wh = W_all[lim]
        Wt = W_all[lim + 1]
        norm = ops.expand_dims(norm, -1)
        t = ops.ones(norm.shape) * 0.5 * (Wh + Wt) 
        soft_input = ops.concatenate((t ** 2, norm ** 2), axis=-1) / self.temp
        softmax_result = ops.softmax(soft_input, axis=-1)
        zw, mw = ops.unstack(softmax_result, axis=-1)
        diff = len(weight.shape) - len(mw.shape)
        for _ in range(diff):
            mw = ops.expand_dims(mw, -1)
        self.mask = mw
        return mw
    
    def get_mask(self, weight):
        if self.is_pretraining:
            self.mask = ops.ones(weight.shape)
            return self.mask
        weight_reshaped = ops.reshape(weight, self.softmax_shape)
        abs_weight_flat = ops.reshape(ops.abs(weight), -1)
        # Do top_k for all weights. Returns sorted array, just use the values on both sides of the threshold (sparsity * size(weight)) to calculate t directly
        all, _ = ops.top_k(abs_weight_flat, ops.size(abs_weight_flat))
        lim = ops.clip(int((1-self.r) * ops.size(abs_weight_flat)), 0, ops.size(abs_weight_flat) -2)
        Wh = all[lim]
        Wt = all[lim + 1]
        t = self.t * (Wh + Wt) 
        soft_input = ops.concatenate((t ** 2, weight_reshaped ** 2), axis=-1) / self.temp
        softmax_result = ops.softmax(soft_input, axis=-1)
        zw, mw = ops.unstack(softmax_result, axis=-1)
        mask = ops.reshape(mw, weight.shape)
        self.mask = mask
        return mask

    def call(self, weight):
        if self.fine_tuning:
            mask = self.mask
        else:
            if self.config["pruning_parameters"]["structured_pruning"]:
                if self.layer_type == "conv":
                    mask = self.get_mask_structured_channel(weight)
                else:
                    mask = self.get_mask_structured_linear(weight)
            else:
                mask = self.get_mask(weight)
        return mask * weight

    def calculate_additional_loss(self):
        return 0
    
    def get_layer_sparsity(self, weight):
        mask = self.mask
        masked_weight_rounded = (mask >= 0.5).float()
        masked_weight = masked_weight_rounded * weight
        return ops.count_nonzero(masked_weight) / ops.size(masked_weight)
    
    def post_epoch_function(self, epoch, total_epochs):
        pass
