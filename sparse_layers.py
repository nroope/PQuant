import os
os.environ["KERAS_BACKEND"] = "torch"
from parser import get_parser
from pruning_methods import get_pruning_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import keras_core as keras


class SparseLayerLinear(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerLinear, self).__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_features)
        self.weight = nn.Parameter(layer.weight.clone())
        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        masked_weight = self.pruning_layer(self.weight)
        return F.linear(x, masked_weight, self.bias)


class SparseLayerConv2d(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerConv2d, self).__init__()
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

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        masked_weight = self.pruning_layer(self.weight)
        return F.conv2d(input=x, weight=masked_weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


class SparseLayerConv1d(nn.Module):
    def __init__(self, config, layer):
        super(SparseLayerConv1d, self).__init__()
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

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        masked_weight = self.pruning_layer(self.weight)
        return F.conv1d(input=x, weight=masked_weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


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


def add_pruning_to_model(module, config):
    for name, layer in module.named_children():
        if isinstance(layer, nn.Linear):
            setattr(module, name, SparseLayerLinear(config, layer))
        elif isinstance(layer, nn.Conv2d):
            setattr(module, name, SparseLayerConv2d(config, layer))
        elif isinstance(layer, nn.Conv1d):
            setattr(module, name, SparseLayerConv1d(config, layer))
        add_pruning_to_model(layer, config)
    return module

def remove_pruning_from_model(module, config):
    for name, layer in module.named_children():
        if isinstance(layer, SparseLayerLinear):
            if config.pruning_method == "pdp": #Find better solution later
                 masked_weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
            else:
                masked_weight = layer.pruning_layer(layer.weight)
            in_features = layer.in_features
            out_features = layer.out_features
            bias_values = layer.bias
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
            getattr(module, name).weight.data = masked_weight
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data = bias_values
        elif isinstance(layer, SparseLayerConv2d):
            if config.pruning_method == "pdp": #Find better solution later
                 masked_weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
            else:
                masked_weight = layer.pruning_layer(layer.weight)
            bias_values = layer.bias
            bias = True if bias_values is not None else False
            setattr(module, name, nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, 
                                           layer.stride, layer.padding, layer.dilation, layer.groups,
                                           bias, layer.padding_mode))
            getattr(module, name).weight.data = masked_weight
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data = bias_values
        remove_pruning_from_model(layer, config)
    return module

def post_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.post_epoch_function(epoch, total_epochs)

def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.pre_epoch_function(epoch, total_epochs)

def post_round_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.post_round_function()

def save_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.save_weights()

def rewind_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.rewind_weights()

def pre_finetune_functions(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.pre_finetune_function()

def post_pretrain_functions(model, config):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
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
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
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
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
            weight_size = layer.weight.numel()
            w = torch.sum(global_weights_below_threshold[idx:idx+weight_size])
            layer.pruning_layer.init_r = w / weight_size
            idx += weight_size


def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                ratio = layer.pruning_layer.get_layer_sparsity(layer.weight)
                total_w += layer.weight.numel()
                remaining_weights += ratio * layer.weight.numel()
        elif isinstance(layer, (nn.Conv2d, nn.Linear)):
             total_w += layer.weight.numel()
             remaining_weights += torch.count_nonzero(layer.weight)
    print(f"Remaining weights: {remaining_weights}/{total_w} = {remaining_weights / total_w}")
    return remaining_weights / total_w

def get_model_losses(model, losses):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                loss = layer.pruning_layer.calculate_additional_loss()                
                losses += loss
    return losses

def test_layer_replacing():
    parser = get_parser()
    args = ["--pruning_method", "dst"]
    config = parser.parse_args(args=args)
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


def test_dst_dstkeras_equals():
    # Test that DST (PyTorch) and DST(Keras) give same output
    from parser import get_parser
    from torchsummary import summary
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    args = ["--pruning_method", "dst"]
    parser = get_parser()
    config = parser.parse_args(args=args)
    model = SingleConvLayer()
    model_keras = SingleConvLayer()
    model.to(device)
    model_keras.to(device)
    model_keras.load_state_dict(model.state_dict())
    test_input = torch.rand(2, 3, 32, 32).to(device)
    summary(model, (3,32,32))
    summary(model_keras, (3,32,32))
    model = add_pruning_to_model(model, config)
    config.pruning_method = "dstkeras"
    model_keras = add_pruning_to_model(model_keras, config)
    
    output = model(test_input)
    output_keras = model_keras(test_input)    
    assert torch.equal(output, output_keras)


if __name__ == "__main__":
    test_layer_replacing()
    test_dst_dstkeras_equals()