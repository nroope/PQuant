import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
from hgq.quantizer import Quantizer
from keras import ops
from keras.layers import Layer
from quantizers import get_fixed_quantizer

from pquant.core.utils import get_backend, get_pruning_layer


# Compressed Layers for PyTorch
class CompressedLayerBase(nn.Module):
    def __init__(self, config, layer, layer_type):
        super().__init__()
        self.f_weight = torch.tensor(config.quantization_parameters.default_fractional_bits)
        self.i_weight = torch.tensor(config.quantization_parameters.default_integer_bits)
        self.f_bias = torch.tensor(config.quantization_parameters.default_fractional_bits)
        self.i_bias = torch.tensor(config.quantization_parameters.default_integer_bits)
        self.weight = nn.Parameter(layer.weight.clone())
        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config.pruning_parameters.pruning_method
        self.overflow = "SAT_SYM" if config.quantization_parameters.use_symmetric_quantization else "SAT"
        self.quantizer = get_fixed_quantizer(overflow_mode=self.overflow)
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous

        self.bias = nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        self.init_weight = self.weight.clone()
        self.pruning_first = config.training_parameters.pruning_first
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_high_granularity_quantization = config.quantization_parameters.use_high_granularity_quantization
        self.enable_pruning = config.pruning_parameters.enable_pruning
        self.hgq_gamma = config.quantization_parameters.hgq_gamma

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
                weight = self.quantizer(weight, k=torch.tensor(1.0), i=self.i_weight, f=self.f_weight, training=True)
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

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
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


# Compressed layers for TF
class CompressedLayerBase(keras.layers.Layer):
    def __init__(self, config, layer, layer_type):
        super().__init__()
        i_bits = config.quantization_parameters.default_integer_bits
        f_bits = config.quantization_parameters.default_fractional_bits
        self.i_weight = ops.convert_to_tensor(i_bits)
        self.f_weight = ops.convert_to_tensor(f_bits)
        self.i_bias = ops.convert_to_tensor(i_bits)
        self.f_bias = ops.convert_to_tensor(f_bits)
        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config.pruning_parameters.pruning_method
        self.overflow = "SAT_SYM" if config.quantization_parameters.use_symmetric_quantization else "SAT"
        self.hgq_gamma = config.quantization_parameters.hgq_gamma

        self.pruning_first = config.training_parameters.pruning_first
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_high_granularity_quantization = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.enable_pruning = config.pruning_parameters.enable_pruning
        self.do_transpose_data = None
        self.weight_transpose = None
        self.data_transpose = None

    def set_quantization_bits(self, i_bits_w, f_bits_w, i_bits_b, f_bits_b):
        self.i_weight = ops.convert_to_tensor(i_bits_w)
        self.f_weight = ops.convert_to_tensor(f_bits_w)
        self.i_bias = ops.convert_to_tensor(i_bits_b)
        self.f_bias = ops.convert_to_tensor(f_bits_b)

    def set_enable_pruning(self, enable_pruning):
        self.enable_pruning = enable_pruning

    def build(self, input_shape):
        super().build(input_shape)
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
                if self.use_bias:
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
                if self.use_bias:
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
        else:
            self.quantizer = get_fixed_quantizer(round_mode="RND", overflow_mode=self.overflow)

    def save_weights(self):
        self.init_weight = self.weight.value

    def rewind_weights(self):
        self.weight.assign(self.init_weight)

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining:
            return 0.0
        loss = (ops.sum(self.hgq_weight.quantizer.i) + ops.sum(self.hgq_weight.quantizer.f)) * self.hgq_gamma
        if self.bias is not None:
            loss += (ops.sum(self.hgq_bias.quantizer.i) + ops.sum(self.hgq_bias.quantizer.f)) * self.hgq_gamma
        return loss

    def handle_transpose(self, x, transpose, do_transpose=False):
        if do_transpose:
            x = ops.transpose(x, transpose)
        return x

    def quantize_i(self, weight, bias):
        if self.enable_quantization:
            if self.use_high_granularity_quantization:
                weight = self.hgq_weight(weight)
                bias = None if bias is None else self.hgq_bias(bias)
            else:
                weight = self.quantizer(
                    weight, k=ops.convert_to_tensor(1.0), i=self.i_weight, f=self.f_weight, training=True
                )
                bias = (
                    None
                    if bias is None
                    else self.quantizer(bias, k=ops.convert_to_tensor(1.0), i=self.i_bias, f=self.f_bias, training=True)
                )
        return weight, bias

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.handle_transpose(weight, self.weight_transpose, True)
            weight = self.pruning_layer(weight)
            weight = self.handle_transpose(weight, self.weight_transpose_back, True)
        return weight

    def prune_and_quantize(self, weight, bias):
        weight = ops.cast(weight, weight.dtype)
        bias = ops.cast(bias, bias.dtype) if bias is not None else None
        if self.pruning_first:
            weight = self.prune(weight)
            weight, bias = self.quantize_i(weight, bias)
        else:
            weight, bias = self.quantize_i(weight, bias)
            weight = self.prune(weight)
        return weight, bias

    def call(self, x):
        return x

    def collect_input(self, x, weight, training):
        collect_x = self.handle_transpose(x, self.data_transpose, self.do_transpose_data)
        weight_channels_first = self.handle_transpose(weight, self.weight_transpose, True)
        self.pruning_layer.collect_input(collect_x, weight_channels_first, training)

    def collect_output(self, x, training):
        collect_x = self.handle_transpose(x, self.data_transpose, self.do_transpose_data)
        self.pruning_layer.collect_output(collect_x, training)


class CompressedLayerDepthwiseConv2dKeras(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.depthwise_regularizer = layer.depthwise_regularizer
        self.use_bias = layer.use_bias
        self.strides = layer.strides
        self.dilation_rate = layer.dilation_rate
        self.padding = layer.padding
        self.kernel_size = layer.kernel_size
        self.bias_shape = layer.bias.shape if layer.use_bias else None
        self.init_bias = layer.bias.value if layer.use_bias else None
        self.weight_shape = layer.kernel.shape
        self.init_weight = layer.kernel.value
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        self.do_transpose_data = layer.data_format == "channels_last"

    def build(self, input_shape):
        self.weight = self.add_weight(
            self.weight_shape, initializer=self.init_weight, trainable=True, regularizer=self.depthwise_regularizer
        )
        self.bias = (
            self.add_weight(self.bias_shape, initializer=self.init_bias, trainable=True)
            if self.bias_shape is not None
            else None
        )
        super().build(input_shape)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.collect_input(x, weight, training)
        x = ops.depthwise_conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.pruning_method == "activation_pruning":
            self.collect_output(x, training)
        return x


class CompressedLayerConv2dKeras(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.kernel_regularizer = layer.kernel_regularizer
        self.filters = layer.filters
        self.use_bias = layer.use_bias
        self.strides = layer.strides
        self.dilation_rate = layer.dilation_rate
        self.padding = layer.padding
        self.kernel_size = layer.kernel_size
        if hasattr(layer, "groups"):
            self.groups = layer.groups
        self.bias_shape = layer.bias.shape if layer.use_bias else None
        self.init_bias = layer.bias.value if layer.use_bias else None
        self.weight_shape = layer.kernel.shape
        self.init_weight = layer.kernel.value
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        self.do_transpose_data = layer.data_format == "channels_last"

    def build(self, input_shape):
        self.weight = self.add_weight(
            self.weight_shape, initializer=self.init_weight, trainable=True, regularizer=self.kernel_regularizer
        )
        self.bias = (
            self.add_weight(self.bias_shape, initializer=self.init_bias, trainable=True)
            if self.bias_shape is not None
            else None
        )
        super().build(input_shape)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.collect_input(x, weight, training)
        x = ops.conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.bias is not None:
            x = ops.add(x, bias)
        if self.pruning_method == "activation_pruning":
            self.collect_output(x, training)
        return x


class CompressedLayerSeparableConv2dKeras(Layer):
    def __init__(self, config, layer):
        super().__init__()
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        layer.kernel = layer.depthwise_kernel
        bias = layer.use_bias
        layer.use_bias = False
        self.depthwise_conv = CompressedLayerDepthwiseConv2dKeras(config, layer, "conv")
        layer.kernel_regularizer = layer.pointwise_regularizer
        layer.kernel_size = 1
        layer.kernel = layer.pointwise_kernel
        layer.use_bias = bias
        self.pointwise_conv = CompressedLayerConv2dKeras(config, layer, "conv")
        self.do_transpose_data = layer.data_format == "channels_last"

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.depthwise_conv(x, training=training)
        x = self.pointwise_conv(x, training=training)
        return x


class CompressedLayerConv1dKeras(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.kernel_regularizer = layer.kernel_regularizer
        self.filters = layer.filters
        self.use_bias = layer.use_bias
        self.strides = layer.strides
        self.dilation_rate = layer.dilation_rate
        self.padding = layer.padding
        self.kernel_size = layer.kernel_size
        self.groups = layer.groups
        self.bias_shape = layer.bias.shape if layer.use_bias else None
        self.init_bias = layer.bias.value if layer.use_bias else None
        self.weight_shape = layer.kernel.shape
        self.init_weight = layer.kernel.value
        self.weight_transpose = (2, 1, 0)
        self.weight_transpose_back = (2, 1, 0)
        self.data_transpose = (0, 2, 1)
        self.do_transpose_data = layer.data_format == "channels_last"

    def build(self, input_shape):
        self.weight = self.add_weight(
            self.weight_shape, initializer=self.init_weight, trainable=True, regularizer=self.kernel_regularizer
        )
        self.bias = (
            self.add_weight(self.bias_shape, initializer=self.init_bias, trainable=True)
            if self.bias_shape is not None
            else None
        )
        super().build(input_shape)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.collect_input(x, weight, training)
        x = ops.conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.bias is not None:
            x = ops.add(x, bias)
        if self.pruning_method == "activation_pruning":
            self.collect_output(x, training)
        return x


class CompressedLayerDenseKeras(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.kernel_regularizer = layer.kernel_regularizer
        self.use_bias = layer.use_bias
        self.units = layer.units
        self.bias_shape = layer.bias.shape if layer.use_bias else None
        self.init_bias = layer.bias.value if layer.use_bias else None
        self.weight_shape = layer.kernel.shape
        self.init_weight = layer.kernel.value
        self.weight_transpose = (1, 0)
        self.weight_transpose_back = (1, 0)
        self.data_transpose = (0, 1)  # Always (BATCH_SIZE, OUT_FEATURES)

    def build(self, input_shape):
        self.weight = self.add_weight(
            self.weight_shape, initializer=self.init_weight, trainable=True, regularizer=self.kernel_regularizer
        )
        self.bias = (
            self.add_weight(self.bias_shape, initializer=self.init_bias, trainable=True)
            if self.bias_shape is not None
            else None
        )
        super().build(input_shape)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.collect_input(x, weight, training)
        x = ops.matmul(x, weight)
        if self.bias is not None:
            x = ops.add(x, bias)
        if self.pruning_method == "activation_pruning":
            self.collect_output(x, training)
        return x


def add_default_layer_quantization_pruning_to_config(model, config):
    backend = get_backend()
    return backend.add_default_layer_quantization_pruning_to_config(model, config)


def add_compression_layers(model, config, input_shape):
    backend = get_backend()
    return backend.add_compression_layers(model, config, input_shape)


def get_layer_keep_ratio(model):
    backend = get_backend()
    return backend.get_layer_keep_ratio(model)


def get_model_losses(model, losses):
    backend = get_backend()
    return backend.get_model_losses(model, losses)


def remove_pruning_from_model(model, config):
    backend = get_backend()
    return backend.remove_pruning_from_model(model, config)


def post_training_prune(model, calibration_data, config):
    from pquant.core.tf_backend import TFBackend
    backend = get_backend()
    t_delta = config.pruning_parameters.t_delta
    config.pruning_parameters.t_start_collecting_batch = 0
    for i in range(t_delta):
        inputs = calibration_data[i]
        if i == 0:
            model = backend.add_compression_layers(model, config, inputs.shape)
            backend.post_pretrain_functions(model, config)
        if isinstance(backend, TFBackend):
            model(inputs, training=True)
        else:
            model(inputs)
    return backend.remove_pruning_from_model(model, config)
