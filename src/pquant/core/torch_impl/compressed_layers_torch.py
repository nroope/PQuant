import torch
import torch.nn as nn
import torch.nn.functional as F
from hgq.quantizer import Quantizer
from quantizers import get_fixed_quantizer
from torch.fx import symbolic_trace

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.utils import get_pruning_layer

import typing 
if typing.TYPE_CHECKING:
    from pquant.core.torch_impl.fit_compress import call_fitcompress 


from keras import ops

class CompressedLayerBase(nn.Module):
    def __init__(self, config, layer, layer_type):
        super().__init__()
        self.f_weight = torch.tensor(config["quantization_parameters"]["default_fractional_bits"])
        self.i_weight = torch.tensor(config["quantization_parameters"]["default_integer_bits"])
        self.f_bias = torch.tensor(config["quantization_parameters"]["default_fractional_bits"])
        self.i_bias = torch.tensor(config["quantization_parameters"]["default_integer_bits"])

        self.weight = nn.Parameter(layer.weight.clone())
        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config["pruning_parameters"]["pruning_method"]
        self.overflow = "SAT_SYM" if config["quantization_parameters"]["use_symmetric_quantization"] else "SAT"
        self.quantizer = get_fixed_quantizer(overflow_mode=self.overflow)
        self.hgq_heterogeneous = config["quantization_parameters"]["hgq_heterogeneous"]

        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.init_weight = self.weight.clone()
        self.pruning_first = config["training_parameters"]["pruning_first"]
        self.enable_quantization = config["quantization_parameters"]["enable_quantization"]
        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.enable_pruning = config["pruning_parameters"]["enable_pruning"]
        self.hgq_gamma = config["quantization_parameters"]["hgq_gamma"]

    def build(self, input_shape):
        if self.use_high_granularity_quantization:
            if self.hgq_heterogeneous:
                self.hgq_weight = Quantizer(
                    k0=1.0,
                    i0=self.i_weight,
                    f0=self.f_weight,
                    round_mode="RND",
                    overflow_mode=self.overflow,
                    q_type="kif",
                    homogeneous_axis=(),
                )
                self.hgq_weight.build(self.weight.shape)
                if self.bias is not None:
                    self.hgq_bias = Quantizer(
                        k0=1.0,
                        i0=self.i_bias,
                        f0=self.f_bias,
                        round_mode="RND",
                        overflow_mode=self.overflow,
                        q_type="kif",
                        homogeneous_axis=(),
                    )
                    self.hgq_bias.build(self.bias.shape)
            else:
                self.hgq_weight = Quantizer(
                    k0=1.0,
                    i0=self.i_weight,
                    f0=self.f_weight,
                    round_mode="RND",
                    overflow_mode=self.overflow,
                    q_type="kif",
                    heterogeneous_axis=(),
                )
                self.hgq_weight.build(self.weight.shape)
                if self.bias is not None:
                    self.hgq_bias = Quantizer(
                        k0=1.0,
                        i0=self.i_bias,
                        f0=self.f_bias,
                        round_mode="RND",
                        overflow_mode=self.overflow,
                        q_type="kif",
                        heterogeneous_axis=(),
                    )
                    self.hgq_bias.build(self.bias.shape)

    def save_weights(self):
        self.init_weight = self.weight.clone()

    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining:
            return 0.0
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
                weight = self.quantizer(weight, k=torch.tensor(1.0), i=self.i_weight, f=self.f_weight, training=False)
                bias = (
                    None
                    if bias is None
                    else self.quantizer(bias, k=torch.tensor(1.0), i=self.i_bias, f=self.f_bias, training=True)
                )
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
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        x = F.linear(x, weight, bias)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x


class CompressedLayerLinear(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.use_fitcompress = config["fitcompress_parameters"]["enable_fitcompress"]
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
        self.use_fitcompress = config["fitcompress_parameters"]["enable_fitcompress"]
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
        self.use_fitcompress = config["fitcompress_parameters"]["enable_fitcompress"]
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


def add_compression_layers_torch(model, config, input_shape):
    model = add_quantized_activations_to_model_layer(model, config)
    # model = add_quantized_activations_to_model_functional(model, config)
    model = add_pruning_to_model(model, config)
    model = disable_pruning_from_layers(model, config)
    model = add_layer_specific_quantization_to_model(model, config)
    model(torch.rand(input_shape, device=next(model.parameters()).device))
    return model


class QuantizedPooling(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        self.f = torch.tensor(config["quantization_parameters"]["default_fractional_bits"])
        self.i = torch.tensor(config["quantization_parameters"]["default_integer_bits"])
        self.overflow = "SAT_SYM" if config["quantization_parameters"]["use_symmetric_quantization"] else "SAT"
        self.config = config
        self.hgq_heterogeneous = config["quantization_parameters"]["hgq_heterogeneous"]
        self.is_pretraining = True
        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.pooling = layer
        self.use_fitcompress = config["fitcompress_parameters"]["enable_fitcompress"]
        self.post_fitcompress_calibration = False
        self.hgq_gamma = config["quantization_parameters"]["hgq_gamma"]
        self.saved_inputs = []

    def build(self, input_shape):
        if self.use_high_granularity_quantization:
            if self.hgq_heterogeneous:
                self.hgq = Quantizer(
                    k0=1.0,
                    i0=self.i,
                    f0=self.f,
                    round_mode="RND",
                    overflow_mode=self.overflow,
                    q_type="kif",
                    homogeneous_axis=(0,),
                )

            else:
                self.hgq = Quantizer(
                    k0=1.0,
                    i0=self.i,
                    f0=self.f,
                    round_mode="RND",
                    overflow_mode=self.overflow,
                    q_type="kif",
                    heterogeneous_axis=(),
                )
            self.hgq.build(input_shape)
        else:
            self.quantizer = get_fixed_quantizer(round_mode="RND", overflow_mode=self.overflow)

    def set_activation_bits(self, i, f):
        self.i = torch.tensor(i)
        self.f = torch.tensor(f)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0
        return (torch.sum(self.hgq.quantizer.i) + torch.sum(self.hgq.quantizer.f)) * self.config["quantization_parameters"][
            "hgq_gamma"
        ]

    def quantize(self, x):
        if not hasattr(self, "hgq") or not hasattr(self, "quantizer"):
            self.build(x.shape)
        if self.use_high_granularity_quantization:
            x = self.hgq(x)
        else:
            if self.use_fitcompress and self.is_pretraining:
                if self.post_fitcompress_calibration:
                    # Save inputs
                    self.saved_inputs.append(x)
                # During FITcompress, we do not use any quantized pooling
                return ops.average_pool(x, pool_size = 1)
            x = self.quantizer(x, k=torch.tensor(1.0), i=self.i, f=self.f, training=True)
        return x

    def forward(self, x):
        x = self.pooling(x)
        return self.quantize(x)


def add_layer_specific_quantization_to_model(module, config):
    for name, layer in module.named_modules():
        if isinstance(layer, CompressedLayerBase):
            if name in config["quantization_parameters"]["layer_specific"]:
                if "weight" in config["quantization_parameters"]["layer_specific"][name]:
                    weight_int_bits = config["quantization_parameters"]["layer_specific"][name]["weight"]["integer_bits"]
                    weight_fractional_bits = config["quantization_parameters"]["layer_specific"][name]["weight"][
                        "fractional_bits"
                    ]
                    layer.i_weight = torch.tensor(weight_int_bits)
                    layer.f_weight = torch.tensor(weight_fractional_bits)
                if "bias" in config["quantization_parameters"]["layer_specific"][name]:
                    bias_int_bits = config["quantization_parameters"]["layer_specific"][name]["bias"]["integer_bits"]
                    bias_fractional_bits = config["quantization_parameters"]["layer_specific"][name]["bias"][
                        "fractional_bits"
                    ]
                    layer.i_bias = torch.tensor(bias_int_bits)
                    layer.f_bias = torch.tensor(bias_fractional_bits)
            layer.build(None)
        elif layer.__class__ in [QuantizedPooling, QuantizedReLU, QuantizedTanh]:
            if name in config["quantization_parameters"]["layer_specific"]:
                i = config["quantization_parameters"]["layer_specific"][name]["integer_bits"]
                f = config["quantization_parameters"]["layer_specific"][name]["fractional_bits"]
                layer.set_activation_bits(i, f)
    return module


def add_quantized_activations_to_model_layer(module, config):
    if not config["quantization_parameters"]["enable_quantization"]:
        return module
    # Replaces ReLU and Tanh layers with quantized versions
    for name, layer in module.named_children():
        i = config["quantization_parameters"]["default_integer_bits"]
        f = config["quantization_parameters"]["default_fractional_bits"]
        if layer.__class__ in [nn.ReLU]:
            # For ReLU, if using default values, add 1 bit since values are unsigned.
            # Otherwise user provides bits. TODO: Find better way to do this
            f = config["quantization_parameters"]["default_fractional_bits"] + 1
            relu = QuantizedReLU(config, i=i, f=f)
            setattr(module, name, relu)
        elif layer.__class__ in [nn.Tanh]:
            tanh = QuantizedTanh(config, i=0.0, f=f)
            setattr(module, name, tanh)
        elif layer.__class__ in [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            new_layer = QuantizedPooling(config, layer)
            setattr(module, name, new_layer)
        else:
            layer = add_quantized_activations_to_model_layer(layer, config)
    return module


def add_quantized_activations_to_model_functional(module, config):
    # Currently not in use. TODO: Fix this
    if config["quantization_parameters"]["use_high_granularity_quantization"]:
        return module
    # Replaces functional activation calls with quantized versions
    traced_model = symbolic_trace(module)
    for node in traced_model.graph.nodes:
        if node.op in ["call_method", "call_function"] and (node.target == "tanh" or "function relu" in str(node.target)):
            with traced_model.graph.inserting_after(node):
                if node.name in config["quantization_parameters"]["layer_specific"]:
                    bits = config["quantization_parameters"]["layer_specific"][node.name]["bits"]
                else:
                    bits = (
                        config["quantization_parameters"]["default_integer_bits"]
                        + config["quantization_parameters"]["default_fractional_bits"]
                        + 1
                    )  # 1 sign bit
                kwargs = {"bits": bits}
                if node.target == "tanh":
                    kwargs["use_real_tanh"] = config["quantization_parameters"]["use_real_tanh"]
                    kwargs["use_symmetric"] = config["quantization_parameters"]["use_symmetric_quantization"]
                    # new_node = traced_model.graph.call_function(quantized_tanh, node.args, kwargs)
                else:
                    kwargs = {"integer_bits": config["quantization_parameters"]["default_integer_bits"], "bits": bits}
                    # new_node = traced_model.graph.call_function(quantized_relu, node.args, kwargs)
                # node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)

    traced_model.graph.lint()
    traced_model.recompile()
    return traced_model


def disable_pruning_from_layers(module, config):
    for name, layer in module.named_modules():
        enable_pruning = name not in config["pruning_parameters"]["disable_pruning_for_layers"]
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
            if config["pruning_parameters"]["pruning_method"] == "pdp":  # Find better solution later
                if config["training_parameters"]["pruning_first"]:
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
            if config["pruning_parameters"]["pruning_method"] == "pdp":  # Find better solution later
                if config["training_parameters"]["pruning_first"]:
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


def post_pretrain_functions(model, config, train_loader = None, loss_func=None):

    if config["fitcompress_parameters"]["enable_fitcompress"]:
        from pquant.core.torch_impl.fit_compress import call_fitcompress
        config, pruning_mask_importance_scores = call_fitcompress(config, model, train_loader, loss_func)

    #idx = 0
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            layer.pruning_layer.post_pre_train_function()
            layer.post_pre_train_function()
            
            #layer.pruning_layer.mask = pruning_mask_importance_scores[idx]
            #idx += 1

        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
            layer.post_pre_train_function()
    if config["pruning_parameters"]["pruning_method"] == "pdp" or (
        config["pruning_parameters"]["pruning_method"] == "wanda"
        and config["pruning_parameters"]["calculate_pruning_budget"]
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
    threshold = global_weight_topk[int((1 - config["pruning_parameters"]["sparsity"]) * global_weight_topk.numel())]
    global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
            weight_size = layer.weight.numel()
            w = torch.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = w / weight_size
            print(f"PDP Layer {layer} target: {layer.pruning_layer.init_r}")
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
            if layer.use_high_granularity_quantization:
                loss += layer.hgq_loss()
            losses += loss
        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
            if layer.use_high_granularity_quantization:
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
        elif layer.__class__ in [nn.Tanh, nn.ReLU, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            config["layer_specific"][name] = {"integer_bits": 0, "fractional_bits": 7}
    return config


def add_default_layer_quantization_pruning_to_config_torch(model, config):
    custom_scheme = create_default_layer_quantization_pruning_config(model)
    config["quantization_parameters"]["layer_specific"] = custom_scheme["layer_specific"]
    config["pruning_parameters"]["disable_pruning_for_layers"] = custom_scheme["disable_pruning_for_layers"]
    return config