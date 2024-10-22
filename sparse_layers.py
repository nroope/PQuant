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
        self.layer = layer
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

    def save_weights(self):
        self.init_weight = self.weight.clone()
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def forward(self, x):
        masked_weight = self.pruning_layer(self.weight)
        return F.conv2d(input=x, weight=masked_weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


##################################### KERAS WIP #####################################
class SparseLayerLinearKeras(keras.layers.Layer):
    def __init__(self, config, layer):
        super(SparseLayerLinearKeras, self).__init__()
        self.layer = layer
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_features)
        self.weight = self.add_weight(shape=(layer.weight.shape), trainable=True)
        self.bias = self.add_weight(shape=layer.bias.shape) if layer.bias is not None else None

    def call(self, x):
        masked_weight = self.pruning_layer(self.weight)
        return F.linear(x, masked_weight, self.bias)


class SparseLayerConv2dKeras(keras.layers.Layer):
    def __init__(self, config, layer):
        super(SparseLayerConv2dKeras, self).__init__()
        self.pruning_layer = get_pruning_layer(config=config, layer=layer, out_size=layer.out_channels)
        self.weight = self.add_weight(shape = layer.weight.shape, trainable=True)
        self.bias = self.add_weight(shape=layer.bias.shape) if layer.bias is not None else None
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.padding = layer.padding
        self.groups = layer.groups

    def call(self, x):
        masked_weight = self.pruning_layer(self.weight)
        return F.conv2d(input=x, weight = masked_weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
#####################################################################################


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


def add_pruning_to_model(model, config):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            setattr(model, name, SparseLayerLinear(config, layer))
        elif isinstance(layer, nn.Conv2d):
            setattr(model, name, SparseLayerConv2d(config, layer))
        add_pruning_to_model(layer, config)
    return model

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
    Selects bottom % weights globally. Then calculates target sparsity for each layer, which will depend on how large % of
    that layer's weights are also in the global bottom % of weights.
    """
    wp = None
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
            if wp is None:
                 wp = layer.weight.flatten()
            else:
                wp = torch.concat((wp, layer.weight.flatten()))            
    # Calculate smallest % of weights globally
    wp, _ = torch.topk(-torch.abs(wp.flatten()), int((1-config.sparsity) * wp.numel()))
    wp = torch.unique(wp)
    wp = wp.unsqueeze(-1)
    for layer in model.modules():
        # Calculate the % of weights in each layer that are also in the global bottom % of weights
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
            weight = layer.weight.flatten()
            w = torch.sum(weight.detach().cpu().apply_(lambda x: x in wp))
            print(w/weight.numel())
            layer.pruning_layer.init_r = w / weight.numel()
            # Split weight tensor into smaller chunks to use less memory
            #splits = get_splits(weight.shape[0])
            #split_length = weight.shape[0] // splits 
            #weight_total = 0
            #for i in range(splits):
            #    subset = weight[i * split_length : (i+1) * split_length]
            #    weight_total += torch.sum(wp == subset)
            #print(weight_total / weight.numel())
            #layer.pruning_layer.init_r = weight_total / weight.numel() 


def get_splits(shape):
    splits = 1
    max_splits = 2 ** 18
    for i in range(1, max_splits + 1):
        if shape % i == 0:
              splits = i
    print(shape, splits, shape // splits)
    return splits


def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                ratio = layer.pruning_layer.get_layer_sparsity(layer.weight)
                total_w += layer.weight.numel()
                remaining_weights += ratio * layer.weight.numel()
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
    args = ["--pruning_method", "str"]
    config = parser.parse_args(args=args)
    linear_model = SingleLinearLayer()
    sparse_linear_model = add_pruning_to_model(linear_model, config)
    assert torch.equal(linear_model.linear.weight, sparse_linear_model.linear.weight)
    assert torch.equal(linear_model.linear.bias, sparse_linear_model.linear.bias)
    conv_model = SingleConvLayer()
    sparse_conv_model = add_pruning_to_model(conv_model, config)
    assert torch.equal(conv_model.conv.weight, sparse_conv_model.conv.weight)
    assert torch.equal(conv_model.conv.bias, sparse_conv_model.conv.bias)
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