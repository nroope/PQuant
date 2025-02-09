import os
os.environ["KERAS_BACKEND"] = "torch"
from pruning_methods import get_pruning_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from parser import parse_cmdline_args
from quantizers import get_fixed_quantizer
from activations_quantizer import QuantizedReLU, QuantizedTanh, quantized_relu, quantized_tanh
from squark.quantizer import Quantizer
import numpy as np
from torch.fx import symbolic_trace




class SparseLayerLinear(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerLinear, self).__init__()
        self.f_weight = torch.tensor(config.default_fractional_bits)
        self.i_weight = torch.tensor(config.default_integer_bits)  
        self.f_bias = torch.tensor(config.default_fractional_bits)
        self.i_bias = torch.tensor(config.default_integer_bits)  

        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.weight = nn.Parameter(layer.weight.clone())
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_features)
        overflow = "SAT_SYM" if config.use_symmetric_quantization else "SAT"
        self.quantizer = get_fixed_quantizer(overflow_mode=overflow)
        #self.hgq = Quantizer(k0=1.0, i0=self.i.item(), f0=self.f.item(), round_mode="TRN", overflow_mode="SAT_SYM", q_type="kif", heterogeneous_axis=()) # DOES NOT DO PRUNING IN GENERAL
        self.hgq_weight = Quantizer(k0=1.0, i0=self.i_weight.item(), f0=self.f_weight.item(), round_mode="TRN", overflow_mode=overflow, q_type="kif")
        self.hgq_weight.build(self.weight.shape)
        if layer.bias is not None:
            self.hgq_bias = Quantizer(k0=1.0, i0=self.i_bias.item(), f0=self.f_bias.item(), round_mode="TRN", overflow_mode=overflow, q_type="kif")
            self.hgq_bias.build(layer.bias.shape)
        self.hgq_gamma = config.hgq_gamma
        
        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.init_weight = self.weight.clone()
        self.pruning_first = config.pruning_first
        self.enable_quantization = config.enable_quantization
        self.use_high_granularity_quantization = config.use_high_granularity_quantization
        self.enable_pruning = config.enable_pruning
        
    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining:
            return 0.
        loss = (torch.sum(self.hgq_weight.quantizer.i) + torch.sum(self.hgq_weight.quantizer.f)) * self.hgq_gamma
        if self.bias is not None:
            loss += (torch.sum(self.hgq_bias.quantizer.i) + torch.sum(self.hgq_bias.quantizer.f)) * self.hgq_gamma
        return loss

    def quantize(self, weight, bias):
        if self.enable_quantization:
            if self.use_high_granularity_quantization:
                weight = self.hgq_weight(weight)
                bias = None if bias is None else self.hgq_bias(bias)
            else:
                weight = self.quantizer(weight, k=torch.tensor(1.0), i=self.i_weight, f=self.f_weight, training=True)
                bias = None if bias is None else self.quantizer(bias, k=torch.tensor(1.0), i=self.i_bias, f=self.f_bias, training=True)
        return weight, bias

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.pruning_layer(weight)
        return weight

    def prune_and_quantize(self, weight, bias):
        if self.pruning_first:
            weight = self.prune(weight)
            weight, bias = self.quantize(weight, bias)
        else:
            weight, bias = self.quantize(weight, bias)
            weight = self.prune(weight)
        return weight, bias
        
    def forward(self, x):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        x = F.linear(x, weight, bias)
        return x

class SparseLayerConv2d(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerConv2d, self).__init__()
        self.f_weight = torch.tensor(config.default_fractional_bits)
        self.i_weight = torch.tensor(config.default_integer_bits)  
        self.f_bias = torch.tensor(config.default_fractional_bits)
        self.i_bias = torch.tensor(config.default_integer_bits)  
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_channels)
        overflow = "SAT_SYM" if config.use_symmetric_quantization else "SAT"
        self.quantizer = get_fixed_quantizer(overflow_mode=overflow)
        self.weight = nn.Parameter(layer.weight.clone())
        self.init_weight = self.weight.clone()
        #self.hgq = Quantizer(k0=1.0, i0=self.i.item(), f0=self.f.item(), round_mode="TRN", overflow_mode="SAT_SYM", q_type="kif", heterogeneous_axis=()) # DOES NOT DO PRUNING IN GENERAL
        self.hgq_weight = Quantizer(k0=1.0, i0=self.i_weight.item(), f0=self.f_weight.item(), round_mode="TRN", overflow_mode=overflow, q_type="kif")
        self.hgq_weight.build(self.weight.shape)
        if layer.bias is not None:
            self.hgq_bias = Quantizer(k0=1.0, i0=self.i_bias.item(), f0=self.f_bias.item(), round_mode="TRN", overflow_mode=overflow, q_type="kif")
            self.hgq_bias.build(layer.bias.shape)
        self.hgq_gamma = config.hgq_gamma

        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.padding = layer.padding
        self.groups = layer.groups
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.padding_mode = layer.padding_mode
        self.pruning_first = config.pruning_first
        self.enable_quantization = config.enable_quantization
        self.use_high_granularity_quantization = config.use_high_granularity_quantization
        self.enable_pruning = config.enable_pruning

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining:
            return 0.
        loss = (torch.sum(self.hgq_weight.quantizer.i) + torch.sum(self.hgq_weight.quantizer.f)) * self.hgq_gamma
        if self.bias is not None:
            loss += (torch.sum(self.hgq_bias.quantizer.i) + torch.sum(self.hgq_bias.quantizer.f)) * self.hgq_gamma
        return loss

    def quantize(self, weight, bias):
        if self.enable_quantization:
            if self.use_high_granularity_quantization:
                weight = self.hgq_weight(weight)
                bias = None if bias is None else self.hgq_bias(bias)
            else:
                weight = self.quantizer(weight, k=torch.tensor(1.0), i=self.i_weight, f=self.f_weight, training=True)
                bias = None if bias is None else self.quantizer(bias, k=torch.tensor(1.0), i=self.i_bias, f=self.f_bias, training=True)
        return weight, bias

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.pruning_layer(weight)
        return weight

    def prune_and_quantize(self, weight, bias):
        if self.pruning_first:
            weight = self.prune(weight)
            weight, bias = self.quantize(weight, bias)
        else:
            weight, bias = self.quantize(weight, bias)
            weight = self.prune(weight)
        return weight, bias

    def forward(self, x):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        x = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

        return x


class SparseLayerConv1d(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerConv1d, self).__init__()
        self.f_weight = torch.tensor(config.default_fractional_bits)
        self.i_weight = torch.tensor(config.default_integer_bits)  
        self.f_bias = torch.tensor(config.default_fractional_bits)
        self.i_bias = torch.tensor(config.default_integer_bits)  
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_channels)
        overflow = "SAT_SYM" if config.use_symmetric_quantization else "SAT"
        self.quantizer = get_fixed_quantizer(overflow_mode=overflow)
        self.weight = nn.Parameter(layer.weight.clone())
        self.init_weight = self.weight.clone()

        self.hgq_weight = Quantizer(k0=1.0, i0=self.i_weight.item(), f0=self.f_weight.item(), round_mode="TRN", overflow_mode=overflow, q_type="kif")
        self.hgq_weight.build(self.weight.shape)
        if layer.bias is not None:
            self.hgq_bias = Quantizer(k0=1.0, i0=self.i_bias.item(), f0=self.f_bias.item(), round_mode="TRN", overflow_mode=overflow, q_type="kif")
            self.hgq_bias.build(layer.bias.shape)
        self.hgq_gamma = config.hgq_gamma

        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.padding = layer.padding
        self.groups = layer.groups
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.padding_mode = layer.padding_mode
        self.pruning_first = config.pruning_first
        self.enable_quantization = config.enable_quantization
        self.use_high_granularity_quantization = config.use_high_granularity_quantization
        self.enable_pruning = config.enable_pruning

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining:
            return 0.
        loss = (torch.sum(self.hgq_weight.quantizer.i) + torch.sum(self.hgq_weight.quantizer.f)) * self.hgq_gamma
        if self.bias is not None:
            loss += (torch.sum(self.hgq_bias.quantizer.i) + torch.sum(self.hgq_bias.quantizer.f)) * self.hgq_gamma
        return loss

    def quantize(self, weight, bias):
        if self.enable_quantization:
            if self.use_high_granularity_quantization:
                weight = self.hgq_weight(weight)
                bias = None if bias is None else self.hgq_bias(bias)
            else:
                weight = self.quantizer(weight, k=torch.tensor(1.0), i=self.i_weight, f=self.f_weight, training=True)
                bias = None if bias is None else self.quantizer(bias, k=torch.tensor(1.0), i=self.i_bias, f=self.f_bias, training=True)
        return weight, bias

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.pruning_layer(weight)
        return weight

    def prune_and_quantize(self, weight, bias):
        if self.pruning_first:
            weight = self.prune(weight)
            weight, bias = self.quantize(weight, bias)
        else:
            weight, bias = self.quantize(weight, bias)
            weight = self.prune(weight)
        return weight, bias

    def forward(self, x):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        x = F.conv1d(input=x, weight=weight, bias=bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

        return x

class SingleLinearLayer(nn.Module):
    def __init__(self):
        super(SingleLinearLayer, self).__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x

class SingleConvLayer(nn.Module):
    def __init__(self):
        super(SingleConvLayer, self).__init__()
        self.conv = nn.Conv2d(3, 3, (3,3))
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        return x


def add_pruning_and_quantization(model, config):
    model = add_quantized_activations_to_model(model, config)
    model = add_pruning_to_model(model, config)
    model = disable_pruning_from_layers(model, config)
    model = add_layer_specific_quantization_to_model(model, config)
    return model

def add_layer_specific_quantization_to_model(module, config):
    for name, layer in module.named_modules():
        if layer.__class__ in [SparseLayerLinear, SparseLayerConv2d, SparseLayerConv1d]:
            if name in config.layer_specific:
                if "weight" in config.layer_specific[name]:
                    weight_int_bits = config.layer_specific[name]["weight"]["integer_bits"]
                    weight_fractional_bits = config.layer_specific[name]["weight"]["fractional_bits"]
                    layer.i_weight = torch.tensor(weight_int_bits)
                    layer.f_weight = torch.tensor(weight_fractional_bits)
                if "bias" in config.layer_specific[name]:
                    bias_int_bits = config.layer_specific[name]["bias"]["integer_bits"]
                    bias_fractional_bits = config.layer_specific[name]["bias"]["fractional_bits"]
                    layer.i_bias = torch.tensor(bias_int_bits)
                    layer.f_bias = torch.tensor(bias_fractional_bits)
    return module
            

def add_quantized_activations_to_model(module, config):
    # Replaces ReLU and Tanh layers with quantized versions
    for name, layer in module.named_children():
        if layer.__class__ in [nn.ReLU]:
            if name in config.layer_specific:
                bits = config.layer_specific[name]["bits"]
            else:
                bits = 8
            relu = QuantizedReLU(bits = float(bits), config=config)
            setattr(module, name, relu)
        elif layer.__class__ in [nn.Tanh]:
            if name in config.layer_specific:
                bits = config.layer_specific[name]["bits"]
            else:
                bits = 8
            tanh = QuantizedTanh(bits = bits, config=config)
            setattr(module, name, tanh)
    if config.use_high_granularity_quantization:
        return module
    # Replaces functional activation calls with quantized versions
    traced_model = symbolic_trace(module)
    for node in traced_model.graph.nodes:
        if node.op in ["call_method", "call_function"] and (node.target == "tanh" or "function relu" in str(node.target)):
            with traced_model.graph.inserting_after(node):
                if node.name in config.layer_specific:
                    bits = config.layer_specific[node.name]["bits"]
                else:
                    bits = config.default_integer_bits + config.default_fractional_bits + 1 # 1 sign bit
                kwargs = {"bits":bits}
                if node.target == "tanh":
                    kwargs["use_real_tanh"] = config.use_real_tanh
                    kwargs["use_symmetric"] = config.use_symmetric_quantization
                    new_node = traced_model.graph.call_function(quantized_tanh, node.args, kwargs)
                else:
                    new_node = traced_model.graph.call_function(quantized_relu, node.args, kwargs)
                node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)

    traced_model.graph.lint()
    traced_model.recompile()
    return traced_model


def disable_pruning_from_layers(module, config):
    for name, layer in module.named_modules():
        enable_pruning = name not in config.disable_pruning_for_layers
        if layer.__class__ in [SparseLayerLinear, SparseLayerConv2d, SparseLayerConv1d] and not enable_pruning:
            print("TRUE")
            layer.enable_pruning = enable_pruning
    return module

def add_pruning_to_model(module, config):
    for name, layer in module.named_children():
        if layer.__class__ is nn.Linear:
            sparse_layer = SparseLayerLinear(config, layer)
            sparse_layer.pruning_layer.build(layer.weight.shape)
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv2d:
            sparse_layer = SparseLayerConv2d(config, layer)
            sparse_layer.pruning_layer.build(layer.weight.shape)
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv1d:
            sparse_layer = SparseLayerConv1d(config, layer)
            sparse_layer.pruning_layer.build(layer.weight.shape)
            setattr(module, name, sparse_layer)
        else:
            add_pruning_to_model(layer, config)
    return module

def remove_pruning_from_model(module, config):
    for name, layer in module.named_children():
        if isinstance(layer, SparseLayerLinear):
            if config.pruning_method == "pdp": #Find better solution later
                if config.pruning_first:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight, bias = layer.quantize(weight, layer.bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            out_features = layer.out_features
            bias_values = bias
            in_features = layer.in_features
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        elif isinstance(layer, SparseLayerConv2d):
            if config.pruning_method == "pdp": #Find better solution later
                if config.pruning_first:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight, bias = layer.quantize(weight, layer.bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            bias_values = bias
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, 
                                           layer.stride, layer.padding, layer.dilation, layer.groups,
                                           bias, layer.padding_mode))
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        elif isinstance(layer, SparseLayerConv1d):
            if config.pruning_method == "pdp": #Find better solution later
                if config.pruning_first:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight, bias = layer.quantize(weight, bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            bias_values = bias
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Conv1d(layer.in_channels, layer.out_channels, layer.kernel_size, 
                                           layer.stride, layer.padding, layer.dilation, layer.groups,
                                           bias, layer.padding_mode))
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        else:
            remove_pruning_from_model(layer, config)
    return module

def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)

def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.pruning_layer.pre_epoch_function(epoch, total_epochs)

def post_round_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.pruning_layer.post_round_function()

def save_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.save_weights()

def rewind_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.rewind_weights()

def pre_finetune_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.pruning_layer.pre_finetune_function()

def post_pretrain_functions(model, config):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                layer.pruning_layer.post_pre_train_function()
    if config.pruning_method == "pdp":
        pdp_setup(model, config)

def pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on 
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = None
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
            if global_weights is None:
                 global_weights = layer.weight.flatten()
            else:
                global_weights = torch.concat((global_weights, layer.weight.flatten()))

    abs_global_weights = torch.abs(global_weights)
    global_weight_topk, _ = torch.topk(abs_global_weights, abs_global_weights.numel())
    threshold = global_weight_topk[int((1-config.sparsity) * global_weight_topk.numel())]
    global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
            weight_size = layer.weight.numel()
            w = torch.sum(global_weights_below_threshold[idx:idx+weight_size])
            layer.pruning_layer.init_r = w / weight_size
            idx += weight_size

@torch.no_grad
def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
            if layer.pruning_first:
                weight = layer.pruning_layer.mask * layer.weight
                weight, bias = layer.quantize(weight, layer.bias)
                total_w += layer.weight.numel()
                rem = torch.count_nonzero(weight)
                remaining_weights += rem
            else:
                weight, bias = layer.quantize(layer.weight, layer.bias)
                weight = layer.pruning_layer.mask * weight
                total_w += weight.numel()
                rem = torch.count_nonzero(weight)
                remaining_weights += rem
        elif isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Linear)):
             total_w += layer.weight.numel()
             remaining_weights += torch.count_nonzero(layer.weight)
    if total_w != 0:
        print(f"Remaining weights: {remaining_weights}/{total_w} = {remaining_weights / total_w}")
        return remaining_weights / total_w
    return 0.

def get_model_losses(model, losses):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                loss = layer.pruning_layer.calculate_additional_loss()          
                if layer.use_high_granularity_quantization:
                    loss += layer.hgq_loss()
                losses += loss
    return losses


def get_layer_weight_uniques(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
                pruned_weight = layer.pruning_layer(layer.weight)
                pruned_q_weight = quantizer(pruned_weight, torch.tensor(1.), layer.i, layer.f, training=True)
                print(np.unique(pruned_q_weight.detach().cpu().numpy().flatten()))
                print(layer.i, layer.f, len(np.unique(pruned_q_weight.detach().cpu().numpy().flatten())))
    return 0.

def test_layer_replacing():
    args = ["--pruning_config_path", "configs/cs/config.yaml", "--model", "resnet20", "--dataset", "cifar10"]
    config = parse_cmdline_args(args=args)
    linear_model_orig = SingleLinearLayer().to("cuda")
    linear_model = SingleLinearLayer()
    linear_model.load_state_dict(linear_model_orig.state_dict())
    sparse_linear_model = add_pruning_to_model(linear_model, config)
    sparse_linear_model.to("cuda")
    assert torch.equal(linear_model_orig.linear.weight, sparse_linear_model.linear.weight)
    assert torch.equal(linear_model_orig.linear.bias, sparse_linear_model.linear.bias)
    desparse_linear_model = remove_pruning_from_model(sparse_linear_model, config)
    assert torch.equal(linear_model_orig.linear.weight, desparse_linear_model.linear.weight)
    assert torch.equal(linear_model_orig.linear.bias, desparse_linear_model.linear.bias)

    conv_model = SingleConvLayer()
    sparse_conv_model = add_pruning_to_model(conv_model, config)
    assert torch.equal(conv_model.conv.weight, sparse_conv_model.conv.weight)
    assert torch.equal(conv_model.conv.bias, sparse_conv_model.conv.bias)
    desparse_linear_model = remove_pruning_from_model(sparse_linear_model, config)
    assert torch.equal(linear_model_orig.linear.weight, desparse_linear_model.linear.weight)
    assert torch.equal(linear_model_orig.linear.bias, desparse_linear_model.linear.bias)
    print("LAYER REPLACING TESTS PASSED")


def create_default_layer_quantization_pruning_config(model):
    config = {"layer_specific":{}, "disable_pruning_for_layers":[]}
    for name, layer in model.named_modules():
        if layer.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            config["layer_specific"][name] = {"weight":{"integer_bits":0, "fractional_bits":7}, 
                                           "bias":{"integer_bits":0, "fractional_bits":7}}
            config["disable_pruning_for_layers"].append(name)
        elif layer.__class__ in [nn.Tanh, nn.ReLU]:
            config["layer_specific"][name] = {"bits":8}
    traced_model = symbolic_trace(model)
    for node in traced_model.graph.nodes:
        if node.op == "call_method" and node.target == "tanh":
            config["layer_specific"][node.name] = {"bits":8}
        elif node.op == "call_function" and "function relu" == str(node.target):
            config["layer_specific"][node.name] = {"bits":8}
    return config

if __name__ == "__main__":
    test_layer_replacing()