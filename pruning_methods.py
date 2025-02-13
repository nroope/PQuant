import torch
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
from keras import ops
import numpy as np
from utils import cosine_sigmoid_decay
import torch
import torch.nn as nn

def get_pruning_layer(config, layer, out_size):
        if config.pruning_method == "dst":
            return DST(config, layer, out_size)
        elif config.pruning_method == "autosparse":
            return AutoSparse(config, layer, out_size)
        elif config.pruning_method == "cs":
            return ContinuousSparsification(config, layer, out_size)
        elif config.pruning_method == "pdp":
            return PDP(config, layer, out_size)
        elif config.pruning_method == "continual_learning":
            return ActivationPruning(config, layer, out_size)
        elif config.pruning_method == "wanda":
            return Wanda(config, layer, out_size)

def get_threshold_size(config, size, weight_shape):
    if config.threshold_type == "layerwise":
        return (1,1)
    elif config.threshold_type == "channelwise":
        return (size, 1)
    elif config.threshold_type == "weightwise":
        return (weight_shape[0], np.prod(weight_shape[1:]))


class Wanda(nn.Module):

    def __init__(self, config, layer, out_size, *args, **kwargs):
            super(Wanda, self).__init__(*args, **kwargs)
            self.config = config
            self.act_type = "relu"
            self.t = 0
            self.layer_type = "linear" if isinstance(layer, nn.Linear) else "conv"
            self.shape = (layer.weight.shape[0], 1)
            if self.layer_type == "conv":
                self.shape = (layer.weight.shape[0], 1, 1, 1)
            self.mask = torch.nn.Parameter(torch.ones(self.shape, requires_grad=False).to(layer.weight.device), requires_grad=False)
            self.inputs = None
            self.total = 0.
            self.weight = layer.weight
            self.done = False
            self.sparsity = self.config.sparsity
            self.is_pretraining = True

    def collect_input(self, x):
        if self.done or self.is_pretraining:
            return
        """
        Accumulates layer inputs until step t_delta, then averages it. 
        Calculates a metric based on weight absolute values and norm of inputs.
        For linear layers, calculate norm over batch dimension.
        For conv layers, take average over batch dimension and calculate norm over flattened kernel_size dimension
        """
        if not self.training or x.shape[0] != self.config.batch_size:
            # Don't collect during validation
            return
        self.t += 1
        self.total += x.shape[0]
        self.inputs = x if self.inputs is None else self.inputs + x
        if self.t % self.config.t_delta == 0:
            inputs_avg = self.inputs / self.total
            self.t = 0
            self.total = 0
            if self.layer_type == "linear":
                norm = inputs_avg.norm(p=2, dim=0)
                metric = self.weight.abs() * norm
                _, sorted_idx = torch.sort(metric, dim=1)
                pruned_idx = sorted_idx[:,:int(self.weight.shape[1] * self.sparsity)]
                self.weight.data = torch.scatter(self.weight, dim=1, index=pruned_idx, src=torch.zeros(pruned_idx.shape).to(self.weight.device))
                self.mask.data = (self.weight != 0).float()
            else:
                inputs_avg = torch.mean(inputs_avg.view(inputs_avg.shape[0], inputs_avg.shape[1], -1), dim=0)
                norm = inputs_avg.norm(p=2, dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                metric = self.weight.abs() * norm
                _, sorted_idx = torch.sort(metric, dim=1)
                pruned_idx = sorted_idx[:, :int(self.weight.shape[1] * self.sparsity)]
                self.weight.data = torch.scatter(self.weight, dim=1, index=pruned_idx, src=torch.zeros(pruned_idx.shape).to(self.weight.device))
                self.mask.data = (self.weight != 0).float()
            self.done = True
            self.inputs = None

    def build(self, weight):
        # Since this is a torch layer, do nothing
        pass

    def forward(self, weight): # Mask is only updated every t_delta step, using collect_output
        self.weight = weight
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


class ActivationPruning(nn.Module):

    def __init__(self, config, layer, out_size, *args, **kwargs):
            super(ActivationPruning, self).__init__(*args, **kwargs)
            self.config = config
            self.act_type = "relu"
            self.t = 0
            self.layer_type = "linear" if isinstance(layer, nn.Linear) else "conv"
            self.shape = (layer.weight.shape[0], 1)
            if self.layer_type == "conv":
                self.shape = (layer.weight.shape[0], 1, 1, 1)
            self.mask = torch.ones(self.shape, requires_grad=False).to(layer.weight.device)
            self.activations = None
            self.total = 0.

    def collect_output(self, output):
        """
        Accumulates values for how often the outputs of the neurons and channels of 
        linear/convolution layer are over 0. Every t_delta steps, uses these values to update 
        the mask to prune those channels and neurons that are active less than a given threshold
        """
        if not self.training:
            # Don't collect during validation
            return
        if self.activations is None:
            # Initialize activations dynamically
            self.activations = torch.zeros(size=output.shape[1:], dtype=output.dtype, device=self.mask.device)
        self.t += 1
        self.total += output.shape[0]
        gt_zero = (output > 0).float()
        gt_zero = torch.sum(gt_zero, dim=0) # Sum over batch, take average during mask update
        self.activations += gt_zero
        if self.t % self.config.t_delta == 0:
            pct_active = self.activations / self.total
            self.t = 0
            self.total = 0
            if self.layer_type == "linear":
                self.mask = (pct_active > self.config.threshold).float().unsqueeze(1)
            else:
                pct_active = pct_active.view(pct_active.shape[0], -1)
                pct_active_avg = torch.mean(pct_active, dim=-1)
                pct_active_above_threshold = (pct_active_avg > self.config.threshold).float()
                self.mask = (pct_active_above_threshold).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.activations *= 0.

    def build(self, weight):
        # Since this is a torch layer, do nothing
        pass

    def forward(self, weight): # Mask is only updated every t_delta step, using collect_output
        return self.mask * weight
    
    def post_pre_train_function(self):
        pass

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

class PDP(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(PDP, self).__init__(*args, **kwargs)
        self.init_r = config.sparsity
        self.r = config.sparsity
        self.temp = config.temperature
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
            self.r = ops.minimum(1., self.config.epsilon * (epoch + 1)) * self.init_r
    
    def post_round_function(self):
        pass

    def get_hard_mask(self, weight):
        if self.fine_tuning:
            return (self.mask >= 0.5).float()
        if self.config.structured_pruning:
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
            if self.config.structured_pruning:
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
        return torch.count_nonzero(masked_weight) / ops.size(masked_weight)
    
    def post_epoch_function(self, epoch, total_epochs):
        pass

class ContinuousSparsification(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(ContinuousSparsification, self).__init__(*args, **kwargs)
        self.config = config
        self.beta = 1
        self.s = self.add_weight(name="threshold", shape=layer.weight.shape, initializer="ones")
        self.s.assign(config.threshold_init * self.s)
        self.s_init = config.threshold_init
        self.final_temp = config.final_temp
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
            scaling = 1. / ops.sigmoid(self.config.threshold_init)
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
        return self.config.threshold_decay * ops.norm(ops.reshape(self.mask, -1), ord=1)
    
    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask()) / ops.size(weight)


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
        self.threshold.assign(config.threshold_init * self.threshold)
        self.alpha = ops.convert_to_tensor(config.alpha, dtype="float32")
        self.g = ops.sigmoid
        self.config = config
        global BACKWARD_SPARSITY
        BACKWARD_SPARSITY = config.backward_sparsity

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
    
    def post_epoch_function(self, epoch, total_epochs, alpha_multiplier, autotune_epochs=0, writer=None, global_step=0):
        # Decay alpha
        if epoch >= autotune_epochs:
            self.alpha *= cosine_sigmoid_decay(epoch - autotune_epochs, total_epochs)
        else:
            self.alpha *= alpha_multiplier
        if epoch == self.config.alpha_reset_epoch:
            self.alpha *= 0.
        if writer is not None:
            writer.write_scalars([(f"Autosparse_alpha", self.alpha, global_step)])


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
        self.mask = torch.ones(layer.weight.shape, requires_grad=False)

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
    
    def get_hard_mask(self, weight):
        return self.mask

    def get_mask(self, weight):
        weight_orig_shape = weight.shape
        weights_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        pre_binarystep_weights = ops.abs(weights_reshaped) - self.threshold
        mask = binary_step(pre_binarystep_weights)
        mask = ops.reshape(mask, weight_orig_shape)
        self.mask = mask
        return mask

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_mask(weight)) / ops.size(weight)
    
    def calculate_additional_loss(self):
        return self.config.alpha * ops.sum(ops.exp(-self.threshold))

    def post_epoch_function(self, epoch, total_epochs):
        pass
    def post_pre_train_function(self):
        pass
    def post_round_function(self):
        pass



############ Torch implementations of Autosparse, CS, DST ############
class AutoSparsePruner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        #x, alpha = input
        mask = torch.relu(x)
        backward_sparsity = 0.5
        x_flat = x.view(-1)
        k = int(x_flat.numel() * backward_sparsity)
        topks, _ = torch.topk(x_flat, k)
        kth_value = topks[-1]
        ctx.save_for_backward(x, alpha, kth_value)
        return mask.view(x.shape)
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, kth_value = ctx.saved_tensors
        grads = torch.ones(x.shape).to("cuda")
        grads[x <= 0] = alpha
        if BACKWARD_SPARSITY:
            grads[x <= kth_value] = 0
        return grads * grad_output, None

class AutoSparseTorch(keras.layers.Layer):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(AutoSparseTorch, self).__init__(*args, **kwargs)
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = nn.Parameter(torch.ones(threshold_size) * config.threshold_init)
        self.alpha = torch.tensor(config.alpha, requires_grad=False).to(device="cuda")
        self.g = torch.sigmoid
        self.prune = AutoSparsePruner.apply
        self.config = config
        global BACKWARD_SPARSITY
        BACKWARD_SPARSITY = config.backward_sparsity

    def forward(self, weight):
        """
        sign(W) * ReLu(X), where X = |W| - sigmoid(threshold), with gradient:
            1 if W > 0 else alpha. Alpha is decayed after each epoch.
        """
        mask = self.get_mask(weight)
        self.mask = mask.view(weight.shape)
        return torch.sign(weight) * mask.view(weight.shape)

    def get_hard_mask(self, weight):
        return self.mask

    def get_mask(self, weight):
        weight_reshaped = weight.view(weight.shape[0], -1)
        w_t = torch.abs(weight_reshaped) - self.g(self.threshold)
        return self.prune(w_t, self.alpha)

    def get_layer_sparsity(self, weight):
        masked_weight = self.get_mask(weight)
        masked_count = ops.count_nonzero(masked_weight)
        return masked_count / ops.size(weight)
    
    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def calculate_additional_loss(*args, **kwargs):
        return 0
    
    def post_epoch_function(self, epoch, total_epochs, alpha_multiplier, autotune_epochs=0, writer=None, global_step=0):
        # Decay alpha
        if epoch >= autotune_epochs:
            self.alpha *= cosine_sigmoid_decay(epoch - autotune_epochs, total_epochs)
        else:
            self.alpha *= alpha_multiplier
        if epoch == self.config.alpha_reset_epoch:
            self.alpha *= 0.
        if writer is not None:
            writer.write_scalars([(f"Autosparse_alpha", self.alpha, global_step)])


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


class DSTTorch(nn.Module):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(DSTTorch, self).__init__(*args, **kwargs)
        threshold_size = get_threshold_size(config, out_size, layer.weight.shape)
        self.threshold = nn.Parameter(torch.zeros(threshold_size))
        self.config = config
        self.step = BinaryStep.apply

    def forward(self, weight):
        """
        ReLu(|W| - T), with gradient:
            2 - 4*|W| if |W| <= 0.4
            0.4           if 0.4 < |W| <= 1
            0             if |W| > 1
        """
        mask = self.get_mask(weight)
        ratio = 1.0 - torch.sum(mask) / mask.numel()
        if ratio >= self.config.max_pruning_pct:
            self.threshold.assign(torch.zeros(self.threshold.shape))
            mask = self.get_mask(weight)
        masked_weight = weight * mask
        return masked_weight
    
    def get_hard_mask(self, weight):
        return self.mask

    def get_mask(self, weight):
        weight_orig_shape = weight.shape
        weights_reshaped = weight.view(weight.shape[0], -1)
        pre_binarystep_weights = torch.abs(weights_reshaped) - self.threshold
        mask = self.step(pre_binarystep_weights)
        mask = mask.view(weight_orig_shape)
        self.mask = mask
        return mask

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def get_layer_sparsity(self, weight):
        return torch.sum(self.get_mask(weight)) / weight.numel()
    
    def calculate_additional_loss(self):
        return self.config.alpha * torch.sum(torch.exp(-self.threshold))

    def post_epoch_function(self, epoch, total_epochs):
        pass
    def post_pre_train_function(self):
        pass
    def post_round_function(self):
        pass

class ContinuousSparsificationTorch(nn.Module):
    def __init__(self, config, layer, out_size, *args, **kwargs):
        super(ContinuousSparsificationTorch, self).__init__(*args, **kwargs)
        self.config = config
        self.beta = torch.tensor(1.)
        self.final_temp = config.final_temp
        self.init_weight = layer.weight.clone()
        self.do_hard_mask = False
        self.mask = None
        self.s_init = torch.tensor(config.threshold_init)
        self.s = nn.Parameter(torch.ones(layer.weight.shape) * self.s_init, requires_grad=True)


    def forward(self, weight):
        self.mask = self.get_mask()
        return self.mask * weight

    def pre_finetune_function(self):
        self.do_hard_mask = True
    
    def get_mask(self):
        if self.do_hard_mask:
            mask = self.get_hard_mask()
            return mask
        else:
            scaling = torch.tensor(1.) / torch.sigmoid(self.s_init)
            mask = torch.nn.functional.sigmoid(self.beta * self.s)
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
        min_beta_s_s0 = torch.minimum(self.beta * self.s, self.s_init)
        self.s.data = min_beta_s_s0
        self.beta = 1

    def calculate_additional_loss(self):
        return self.config.threshold_decay * torch.norm(self.mask.view(-1), p=1)
    
    def get_layer_sparsity(self, weight):
        return torch.sum(self.get_hard_mask()) / weight.numel()



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