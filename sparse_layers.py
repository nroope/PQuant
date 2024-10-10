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


def call_post_epoch_function(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.post_epoch_function(epoch, total_epochs)

def call_post_round_function(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.post_round_function()

def call_save_weights_function(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.save_weights()

def call_rewind_weights_function(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.rewind_weights()

def call_pre_finetune_function(model):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                layer.pruning_layer.pre_finetune_function()

def get_layer_keep_ratio(model, ratios):
    for layer in model.modules():
        if isinstance(layer, (SparseLayerConv2d, SparseLayerLinear)):
                ratio = layer.pruning_layer.get_layer_sparsity(layer.weight)
                ratios = torch.concat((ratios, torch.unsqueeze(ratio, 0)))
    return ratios


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