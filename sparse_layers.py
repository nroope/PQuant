import os
os.environ["KERAS_BACKEND"] = "torch"
from pruning_methods import get_pruning_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from parser import parse_cmdline_args
from quantizers import get_fixed_quantizer
import numpy as np

quantizer = get_fixed_quantizer(overflow_mode="SAT")

class SparseLayerLinear(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerLinear, self).__init__()
        self.f = torch.tensor(config.default_fractional_bits)
        self.i = torch.tensor(config.default_integer_bits)  
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_features)
        self.weight = nn.Parameter(layer.weight.clone())
        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.init_weight = self.weight.clone()
        self.pruning_first = config.pruning_first

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        if self.pruning_first:
            weight = self.pruning_layer(self.weight)
            weight = quantizer(weight, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            bias = None if self.bias is None else quantizer(self.bias, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            x = F.linear(x, weight, bias)
        else:
            weight = quantizer(self.weight, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            bias = None if self.bias is None else quantizer(self.bias, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            weight = self.pruning_layer(weight)
            x = F.linear(x, weight, bias)
        return x

class SparseLayerConv2d(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerConv2d, self).__init__()
        self.f = torch.tensor(config.default_fractional_bits)
        self.i = torch.tensor(config.default_integer_bits)  
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_channels)
        self.weight = nn.Parameter(layer.weight.clone())
        self.init_weight = self.weight.clone()
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

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        if self.pruning_first:
            weight = self.pruning_layer(self.weight)
            weight = quantizer(weight, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            bias = None if self.bias is None else quantizer(self.bias, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            x = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            weight = quantizer(self.weight, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            bias = None if self.bias is None else quantizer(self.bias, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            weight = self.pruning_layer(weight)
            x = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

        return x


class SparseLayerConv1d(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerConv1d, self).__init__()
        self.f = torch.tensor(config.default_fractional_bits)
        self.i = torch.tensor(config.default_integer_bits) 
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_channels)
        self.weight = nn.Parameter(layer.weight.clone())
        self.init_weight = self.weight.clone()
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


    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        if self.pruning_first:
            weight = self.pruning_layer(self.weight)
            weight = quantizer(weight, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            bias = None if self.bias is None else quantizer(self.bias, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            x = F.conv1d(input=x, weight=weight, bias=bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            weight = quantizer(self.weight, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            bias = None if self.bias is None else quantizer(self.bias, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
            weight = self.pruning_layer(weight)
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


def add_layer_specific_quantization_to_model(module, config):
    for name, layer in module.named_modules():
        if layer.__class__ in [SparseLayerLinear, SparseLayerConv2d, SparseLayerConv1d]:
            if name in config.layer_specific:
                int_bits, float_bits = config.layer_specific[name]["integer_bits"], config.layer_specific[name]["float_bits"]
                layer.i = torch.tensor(int_bits)
                layer.f = torch.tensor(float_bits)
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
                    weight = quantizer(weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                else:
                    weight = quantizer(layer.weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                if config.pruning_first:
                    weight = layer.pruning_layer(layer.weight)
                    weight = quantizer(weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                else:
                    weight = quantizer(layer.weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                    weight = layer.pruning_layer(weight)
            out_features = layer.out_features
            bias_values = layer.bias
            in_features = layer.in_features
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(layer.bias.data)
        elif isinstance(layer, SparseLayerConv2d):
            if config.pruning_method == "pdp": #Find better solution later
                if config.pruning_first:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight = quantizer(weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                else:
                    weight = quantizer(layer.weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight

            else:
                if config.pruning_first:
                    weight = layer.pruning_layer(layer.weight)
                    weight = quantizer(weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                else:
                    weight = quantizer(layer.weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                    weight = layer.pruning_layer(weight)
            bias_values = layer.bias
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, 
                                           layer.stride, layer.padding, layer.dilation, layer.groups,
                                           bias, layer.padding_mode))
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(layer.bias.data)
        elif isinstance(layer, SparseLayerConv1d):
            if config.pruning_method == "pdp": #Find better solution later
                if config.pruning_first:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight = quantizer(weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                else:
                    weight = quantizer(layer.weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight

            else:
                if config.pruning_first:
                    weight = layer.pruning_layer(layer.weight)
                    weight = quantizer(weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                else:
                    weight = quantizer(layer.weight, k=torch.tensor(1.), i=layer.i, f=layer.f, training=True)
                    weight = layer.pruning_layer(weight)
            bias_values = layer.bias
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Conv1d(layer.in_channels, layer.out_channels, layer.kernel_size, 
                                           layer.stride, layer.padding, layer.dilation, layer.groups,
                                           bias, layer.padding_mode))
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(layer.bias.data)
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


def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerConv1d, SparseLayerLinear)):
            if layer.pruning_first:
                weight = layer.pruning_layer.mask * layer.weight
                weight = quantizer(weight, k=torch.tensor(1.0), i=layer.i, f=layer.f)
                total_w += layer.weight.numel()
                rem = torch.count_nonzero(weight)
                remaining_weights += rem
            else:
                weight = quantizer(layer.weight, k=torch.tensor(1.0), i=layer.i, f=layer.f)
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


if __name__ == "__main__":
    test_layer_replacing()