import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.fx import symbolic_trace

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.utils import get_pruning_layer

if typing.TYPE_CHECKING:
    from pquant.core.torch_impl.fit_compress import call_fitcompress  # noqa: 401

from keras import ops

from pquant.core.quantizer_functions import create_quantizer


class CompressedLayerBase(nn.Module):
    def __init__(self, config, layer, layer_type):
        super().__init__()
        self.i_weight = self.i_bias = self.i_input = self.i_output = torch.tensor(
            config.quantization_parameters.default_integer_bits
        )
        self.f_weight = self.f_bias = self.f_input = self.f_output = torch.tensor(
            config.quantization_parameters.default_fractional_bits
        )

        self.weight = nn.Parameter(layer.weight.clone())
        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config.pruning_parameters.pruning_method
        
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.use_input_quantizer = False
        self.use_output_quantizer = False

        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.init_weight = self.weight.clone()
        self.pruning_first = config.training_parameters.pruning_first
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_pruning = config.pruning_parameters.enable_pruning
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.hgq_gamma = config.quantization_parameters.hgq_gamma

    def build(self):
        # Build function to delay quantizer creation until after custom i,f bits have been set
        self.weight_quantizer = PyTorchQuantizer(
            torch.tensor(1.0),
            self.i_weight,
            self.f_weight,
            self.overflow,
            self.round_mode,
            self.use_hgq,
            False,
            self.hgq_gamma,
        )

        if self.bias is not None:
            self.bias_quantizer = PyTorchQuantizer(
                torch.tensor(1.0),
                self.i_bias,
                self.f_bias,
                self.overflow,
                self.round_mode,
                self.use_hgq,
                False,
                self.hgq_gamma,
            )
        if self.use_input_quantizer:
            self.input_quantizer = PyTorchQuantizer(
                torch.tensor(1.0),
                self.i_input,
                self.f_input,
                self.overflow,
                self.round_mode,
                self.use_hgq,
                True,
                self.hgq_gamma,
            )
        if self.use_output_quantizer:
            self.output_quantizer = PyTorchQuantizer(
                torch.tensor(1.0),
                self.i_output,
                self.f_output,
                self.overflow,
                self.round_mode,
                self.use_hgq,
                True,
                self.hgq_gamma,
            )

    def save_weights(self):
        self.init_weight = self.weight.clone()

    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining or not self.use_hgq:
            return 0.0
        loss = (
            torch.sum(self.weight_quantizer.quantizer.quantizer.i) + torch.sum(self.weight_quantizer.quantizer.quantizer.f)
        ) * self.hgq_gamma
        if self.bias is not None:
            loss += (
                torch.sum(self.bias_quantizer.quantizer.quantizer.i) + torch.sum(self.bias_quantizer.quantizer.quantizer.f)
            ) * self.hgq_gamma
        if self.use_input_quantizer:
            loss += (
                torch.sum(self.input_quantizer.quantizer.quantizer.i) + torch.sum(self.input_quantizer.quantizer.quantizer.f)
            ) * self.hgq_gamma
        if self.use_output_quantizer:
            loss += (
                torch.sum(self.output_quantizer.quantizer.quantizer.i)
                + torch.sum(self.output_quantizer.quantizer.quantizer.f)
            ) * self.hgq_gamma
        return loss

    def quantize(self, weight, bias):
        if self.enable_quantization:
            weight = self.weight_quantizer(weight)
            bias = None if bias is None else self.bias_quantizer(bias)
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

    def pre_forward(self, weight, bias, x):
        if self.use_input_quantizer and not self.use_fitcompress and not self.pruning_layer.is_pretraining:
            x = self.input_quantizer(x)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        weight, bias = self.prune_and_quantize(weight, bias)
        return weight, bias, x

    def post_forward(self, x):
        if self.use_output_quantizer and not self.use_fitcompress and not self.pruning_layer.is_pretraining:
            x = self.output_quantizer(x)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x

    def forward(self, x):
        weight, bias, x = self.pre_forward(self.weight, self.bias, x)
        x = F.linear(x, weight, bias)
        x = self.post_forward(x)
        return x


class CompressedLayerLinear(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.is_pretraining = True

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, x):
        if self.use_fitcompress and self.is_pretraining:
            weight, bias = self.weight, self.bias
        else:
            weight, bias = self.prune_and_quantize(self.weight, self.bias)

        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        x = F.linear(x, weight, bias)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x


class CompressedLayerConv2d(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.padding = layer.padding
        self.groups = layer.groups
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.padding_mode = layer.padding_mode
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.is_pretraining = True

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, x):
        if self.use_fitcompress and self.is_pretraining:
            weight, bias = self.weight, self.bias
        else:
            weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, weight, self.training)
        x = F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x


class CompressedLayerConv1d(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)

        self.stride = layer.stride
        self.dilation = layer.dilation
        self.padding = layer.padding
        self.groups = layer.groups
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.padding_mode = layer.padding_mode
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.is_pretraining = True

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, x):
        if self.use_fitcompress and self.is_pretraining:
            weight, bias = self.weight, self.bias
        else:
            weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        x = F.conv1d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x


def add_compression_layers_torch(model, config, input_shape, device="cuda"):
    model = add_quantized_activations_to_model_layer(model, config)
    # model = add_quantized_activations_to_model_functional(model, config)
    model = add_pruning_to_model(model, config)
    model = disable_pruning_from_layers(model, config)
    model = add_layer_specific_quantization_to_model(model, config)
    model.to(device)
    model(torch.rand(input_shape, device=next(model.parameters()).device))
    return model


class QuantizedPooling(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        self.f = torch.tensor(config.quantization_parameters.default_fractional_bits)
        self.i = torch.tensor(config.quantization_parameters.default_integer_bits)
        self.overflow = "SAT_SYM" if config.quantization_parameters.use_symmetric_quantization else "SAT"
        self.config = config
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.is_pretraining = True

        self.overflow = config.quantization_parameters.overflow
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.pooling = layer
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.hgq_gamma = config.quantization_parameters.hgq_gamma

    def build(self, input_shape):
        self.quantizer = PyTorchQuantizer(
            k=torch.tensor(1.0),
            i=self.i,
            f=self.f,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        if self.use_hgq:
            self.quantizer.quantizer.build(input_shape)

    def set_bits(self, i, f):
        self.i = torch.tensor(i)
        self.f = torch.tensor(f)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0

        return (torch.sum(self.quantizer.quantizer.i) + torch.sum(self.quantizer.quantizer.f)) * self.config.quantization_parameters.hgq_gamma

    def quantize(self, x):
        if not hasattr(self, "quantizer"):
            self.build(x.shape)
        if self.use_fitcompress and self.is_pretraining:
            if self.post_fitcompress_calibration:
                # Save inputs
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized pooling
            return ops.average_pool(x, pool_size=1)
        x = self.quantizer(x)
        return x

    def forward(self, x):
        x = self.pooling(x)
        return self.quantize(x)


class PQBatchNorm2d(nn.BatchNorm2d):

    def __init__(
        self,
        config,
        num_features: int,
        eps: float = 1e-5,
        momentum: typing.Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        self.f = torch.tensor(config["quantization_parameters"]["default_fractional_bits"])
        self.i = torch.tensor(config["quantization_parameters"]["default_integer_bits"])
        self.overflow = config["quantization_parameters"]["overflow"]
        self.round_mode = config["quantization_parameters"]["round_mode"]
        self.config = config
        self.parameter_quantizer = PyTorchQuantizer(
            k=torch.tensor(1.0),
            i=self.i,
            f=self.f,
            round_mode=self.round_mode,
            overflow=self.overflow,
            is_data=False,
            is_heterogeneous=False,
        )
        self._weight = nn.Parameter(self.weight.clone())
        self._bias = nn.Parameter(self.bias.clone())
        del self._parameters["weight"]
        del self._parameters["bias"]

    def set_bits(self, i, f):
        self.i = torch.tensor(i)
        self.f = torch.tensor(f)

    @property
    def weight(self):
        return self.parameter_quantizer(self._weight)

    @property
    def bias(self):
        return self.parameter_quantizer(self._bias)

    def set_quantization_bits(self, i, f):
        self.i = torch.tensor(i)
        self.f = torch.tensor(f)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class PyTorchQuantizer(nn.Module):
    # HGQ quantizer wrapper
    def __init__(self, k, i, f, overflow, round_mode, is_heterogeneous, is_data, hgq_gamma=0):
        super().__init__()
        self.k = 1.0
        self.i = i
        self.f = f
        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.quantizer = create_quantizer(self.k, self.i, self.f, overflow, round_mode, is_heterogeneous, is_data)
        self.is_pretraining = False
        self.hgq_gamma = hgq_gamma

    def post_pretrain(self):
        self.is_pretraining = True

    def forward(self, x):
        if self.use_hgq:
            x = self.quantizer(x)
        else:
            x = self.quantizer(x, k=self.k, i=self.i, f=self.f)
        return x

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = (torch.sum(self.quantizer.quantizer.i) + torch.sum(self.quantizer.quantizer.f)) * self.hgq_gamma
        return loss


def add_layer_specific_quantization_to_model(module, config):
    for name, layer in module.named_modules():
        if isinstance(layer, CompressedLayerBase):
            if name in config.quantization_parameters.layer_specific:
                if "weight" in config.quantization_parameters.layer_specific[name]:
                    weight_int_bits = config.quantization_parameters.layer_specific[name]["weight"]["integer_bits"]
                    weight_fractional_bits = config.quantization_parameters.layer_specific[name]["weight"]["fractional_bits"]
                    layer.i_weight = torch.tensor(weight_int_bits)
                    layer.f_weight = torch.tensor(weight_fractional_bits)
                if "bias" in config.quantization_parameters.layer_specific[name]:
                    bias_int_bits = config.quantization_parameters.layer_specific[name]["bias"]["integer_bits"]
                    bias_fractional_bits = config.quantization_parameters.layer_specific[name]["bias"]["fractional_bits"]
                    layer.i_bias = torch.tensor(bias_int_bits)
                    layer.f_bias = torch.tensor(bias_fractional_bits)
            layer.build(None)
        elif layer.__class__ in [PQBatchNorm2d, QuantizedPooling, QuantizedReLU, QuantizedTanh]:
            if name in config.quantization_parameters.layer_specific:
                i = config.quantization_parameters.layer_specific[name]["integer_bits"]
                f = config.quantization_parameters.layer_specific[name]["fractional_bits"]
                layer.set_bits(i, f)
    return module


def add_quantized_activations_to_model_layer(module, config):
    if not config.quantization_parameters.enable_quantization:
        return module
    # Replaces ReLU and Tanh layers with quantized versions
    for name, layer in module.named_children():
        i = config.quantization_parameters.default_integer_bits
        f = config.quantization_parameters.default_fractional_bits
        if layer.__class__ in [nn.ReLU]:
            # For ReLU, if using default values, add 1 bit since values are unsigned.
            # Otherwise user provides bits. TODO: Find better way to do this
            f = config.quantization_parameters.default_fractional_bits + 1
            relu = QuantizedReLU(config, i=i, f=f)
            setattr(module, name, relu)
        elif layer.__class__ in [nn.Tanh]:
            tanh = QuantizedTanh(config, i=0.0, f=f)
            setattr(module, name, tanh)
        elif layer.__class__ in [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            new_layer = QuantizedPooling(config, layer)
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.BatchNorm2d:
            new_layer = PQBatchNorm2d(
                config,
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
            )
            setattr(module, name, new_layer)
        else:
            layer = add_quantized_activations_to_model_layer(layer, config)
    return module


def add_quantized_activations_to_model_functional(module, config):
    # Currently not in use. TODO: Fix this
    if config.quantization_parameters.use_high_granularity_quantization:
        return module
    # Replaces functional activation calls with quantized versions
    traced_model = symbolic_trace(module)
    for node in traced_model.graph.nodes:
        if node.op in ["call_method", "call_function"] and (node.target == "tanh" or "function relu" in str(node.target)):
            with traced_model.graph.inserting_after(node):
                if node.name in config.quantization_parameters.layer_specific:
                    bits = config.quantization_parameters.layer_specific[node.name]["bits"]
                else:
                    bits = (
                        config.quantization_parameters.default_integer_bits
                        + config.quantization_parameters.default_fractional_bits
                        + 1
                    )  # 1 sign bit
                kwargs = {"bits": bits}
                if node.target == "tanh":
                    kwargs["use_real_tanh"] = config.quantization_parameters.use_real_tanh
                    kwargs["use_symmetric"] = config.quantization_parameters.use_symmetric_quantization
                    # new_node = traced_model.graph.call_function(quantized_tanh, node.args, kwargs)
                else:
                    kwargs = {"integer_bits": config.quantization_parameters.default_integer_bits, "bits": bits}
                    # new_node = traced_model.graph.call_function(quantized_relu, node.args, kwargs)
                # node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)

    traced_model.graph.lint()
    traced_model.recompile()
    return traced_model


def disable_pruning_from_layers(module, config):
    for name, layer in module.named_modules():
        enable_pruning = name not in config.pruning_parameters.disable_pruning_for_layers
        if layer.__class__ in [CompressedLayerLinear, CompressedLayerConv2d, CompressedLayerConv1d] and not enable_pruning:
            layer.enable_pruning = enable_pruning
    return module


def add_pruning_to_model(module, config):
    for name, layer in module.named_children():
        if layer.__class__ is nn.Linear:
            sparse_layer = CompressedLayerLinear(config, layer, "linear")
            sparse_layer.pruning_layer.build(layer.weight.shape)
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv2d:
            sparse_layer = CompressedLayerConv2d(config, layer, "conv")
            sparse_layer.pruning_layer.build(layer.weight.shape)
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv1d:
            sparse_layer = CompressedLayerConv1d(config, layer, "conv")
            sparse_layer.pruning_layer.build(layer.weight.shape)
            setattr(module, name, sparse_layer)
        else:
            add_pruning_to_model(layer, config)
    return module


def remove_pruning_from_model_torch(module, config):
    for name, layer in module.named_children():
        if isinstance(layer, CompressedLayerLinear):
            if config.pruning_parameters.pruning_method == "pdp":  # Find better solution later
                if config.training_parameters.pruning_first:
                    weight = layer.weight
                    if layer.enable_pruning:
                        weight = layer.pruning_layer.get_hard_mask(weight) * weight
                    weight, bias = layer.quantize(weight, layer.bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    if layer.enable_pruning:
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
        elif isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d)):
            if config.pruning_parameters.pruning_method == "pdp":  # Find better solution later
                if config.training_parameters.pruning_first:
                    weight = layer.weight
                    if layer.enable_pruning:
                        weight = layer.pruning_layer.get_hard_mask(weight) * weight
                    weight, bias = layer.quantize(weight, layer.bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    if layer.enable_pruning:
                        weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            bias_values = bias
            bias = True if bias_values is not None else False
            conv = nn.Conv2d if isinstance(layer, CompressedLayerConv2d) else nn.Conv1d
            setattr(
                module,
                name,
                conv(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.groups,
                    bias,
                    layer.padding_mode,
                ),
            )
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        else:
            remove_pruning_from_model_torch(layer, config)
    return module


def call_post_round_functions(model, rewind, rounds, r):
    if rewind == "round":
        rewind_weights_functions(model)
    elif rewind == "post-ticket-search" and r == rounds - 1:
        rewind_weights_functions(model)
    else:
        post_round_functions(model)


def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.save_weights()


def rewind_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.rewind_weights()


def pre_finetune_functions(model):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config, train_loader=None, loss_func=None):

    if config.fitcompress_parameters.enable_fitcompress:
        from pquant.core.torch_impl.fit_compress import call_fitcompress

        config, pruning_mask_importance_scores = call_fitcompress(config, model, train_loader, loss_func)

    # idx = 0
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.pruning_layer.post_pre_train_function()
            layer.post_pre_train_function()

            # layer.pruning_layer.mask = pruning_mask_importance_scores[idx]
            # idx += 1

        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
            layer.post_pre_train_function()
    if config.pruning_parameters.pruning_method == "pdp" or (
        config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
    ):
        # pass
        pdp_setup(model, config)


def pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = None
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            if global_weights is None:
                global_weights = layer.weight.flatten()
            else:
                global_weights = torch.concat((global_weights, layer.weight.flatten()))

    abs_global_weights = torch.abs(global_weights)
    global_weight_topk, _ = torch.topk(abs_global_weights, abs_global_weights.numel())
    threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * global_weight_topk.numel())]
    global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            weight_size = layer.weight.numel()
            w = torch.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = w / weight_size
            layer.pruning_layer.sparsity = w / weight_size  # Wanda
            idx += weight_size


@torch.no_grad
def get_layer_keep_ratio_torch(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            if layer.pruning_first:
                weight = layer.weight
                if layer.enable_pruning:
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
                weight, bias = layer.quantize(weight, layer.bias)
                total_w += weight.numel()
                rem = torch.count_nonzero(weight)
                remaining_weights += rem
            else:
                weight, bias = layer.quantize(layer.weight, layer.bias)
                if layer.enable_pruning:
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
                total_w += weight.numel()
                rem = torch.count_nonzero(weight)
                remaining_weights += rem
        elif layer.__class__ in (nn.Conv2d, nn.Conv1d, nn.Linear):
            total_w += layer.weight.numel()
            remaining_weights += torch.count_nonzero(layer.weight)
    if total_w != 0:
        return remaining_weights / total_w
    return 0.0


def get_model_losses_torch(model, losses):
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            loss = layer.pruning_layer.calculate_additional_loss()
            if layer.use_hgq:
                loss += layer.hgq_loss()
            losses += loss
        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
            if layer.use_hgq:
                losses += layer.hgq_loss()
    return losses


def create_default_layer_quantization_pruning_config(model):
    config = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for name, layer in model.named_modules():
        if layer.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            if layer.bias is None:
                config["layer_specific"][name] = {"weight": {"integer_bits": 0, "fractional_bits": 7}}
            else:
                config["layer_specific"][name] = {
                    "weight": {"integer_bits": 0, "fractional_bits": 7},
                    "bias": {"integer_bits": 0, "fractional_bits": 7},
                }
            config["disable_pruning_for_layers"].append(name)
        elif layer.__class__ in [nn.BatchNorm2d, nn.Tanh, nn.ReLU, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            config["layer_specific"][name] = {"integer_bits": 0, "fractional_bits": 7}
    return config


def add_default_layer_quantization_pruning_to_config_torch(model, config):
    custom_scheme = create_default_layer_quantization_pruning_config(model)
    config.quantization_parameters.layer_specific = custom_scheme["layer_specific"]
    config.pruning_parameters.disable_pruning_for_layers = custom_scheme["disable_pruning_for_layers"]
    return config
