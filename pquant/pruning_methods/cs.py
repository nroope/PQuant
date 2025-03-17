import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
from keras import ops


class ContinuousSparsification(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(ContinuousSparsification, self).__init__(*args, **kwargs)
        self.config = config
        self.beta = 1
        self.s = self.add_weight(name="threshold", shape=layer.weight.shape, initializer="ones")
        self.s.assign(config["pruning_parameters"]["threshold_init"] * self.s)
        self.s_init = config["pruning_parameters"]["threshold_init"]
        self.final_temp = config["pruning_parameters"]["final_temp"] 
        self.init_weight = layer.weight.clone()
        self.do_hard_mask = False
        self.mask = None

    def call(self, weight):
        self.mask = self.get_mask()
        return self.mask * weight

    def pre_finetune_function(self):
        self.do_hard_mask = True
    
    def get_mask(self):
        if self.do_hard_mask:
            mask = self.get_hard_mask()
            return mask
        else:
            scaling = 1. / ops.sigmoid(self.config["pruning_parameters"]["threshold_init"])
            mask = ops.sigmoid(self.beta * self.s) 
            mask = mask * scaling
            return mask
    
    def post_pre_train_function(self):
        pass

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def post_epoch_function(self, epoch, total_epochs):
        self.beta *= (self.final_temp**(1/(total_epochs - 1)))

    def get_hard_mask(self):
        return (self.s > 0).float()

    def post_round_function(self):
        min_beta_s_s0 = ops.minimum(self.beta * self.s, self.s_init)
        self.s.assign(min_beta_s_s0)
        self.beta = 1

    def calculate_additional_loss(self):
        return self.config["pruning_parameters"]["threshold_decay"] * ops.norm(ops.reshape(self.mask, -1), ord=1)
    
    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask()) / ops.size(weight)

