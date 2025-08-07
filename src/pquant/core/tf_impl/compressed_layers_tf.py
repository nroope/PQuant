import keras
from hgq.quantizer import Quantizer
from keras import ops
from keras.layers import (
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Layer,
    ReLU,
    SeparableConv2D,
)
from quantizers import get_fixed_quantizer

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.utils import get_pruning_layer


class CompressedLayerBase(keras.layers.Layer):
    def __init__(self, config, layer, layer_type):
        super().__init__()
        i_bits = config["quantization_parameters"]["default_integer_bits"]
        f_bits = config["quantization_parameters"]["default_fractional_bits"]
        self.i_weight = ops.convert_to_tensor(i_bits)
        self.f_weight = ops.convert_to_tensor(f_bits)
        self.i_bias = ops.convert_to_tensor(i_bits)
        self.f_bias = ops.convert_to_tensor(f_bits)
        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config["pruning_parameters"]["pruning_method"]
        self.overflow = "SAT_SYM" if config["quantization_parameters"]["use_symmetric_quantization"] else "SAT"
        self.hgq_gamma = config["quantization_parameters"]["hgq_gamma"]

        self.pruning_first = config["training_parameters"]["pruning_first"]
        self.enable_quantization = config["quantization_parameters"]["enable_quantization"]
        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.hgq_heterogeneous = config["quantization_parameters"]["hgq_heterogeneous"]
        self.enable_pruning = config["pruning_parameters"]["enable_pruning"]
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


class QuantizedPooling(keras.layers.Layer):
    def __init__(self, config, layer):
        super().__init__()
        self.i = ops.convert_to_tensor(config["quantization_parameters"]["default_integer_bits"])
        self.f = ops.convert_to_tensor(config["quantization_parameters"]["default_fractional_bits"])

        self.is_pretraining = True

        self.overflow = "SAT_SYM" if config["quantization_parameters"]["use_symmetric_quantization"] else "SAT"
        self.hgq_gamma = config["quantization_parameters"]["hgq_gamma"]

        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.hgq_heterogeneous = config["quantization_parameters"]["hgq_heterogeneous"]
        self.pool_size = layer.pool_size
        self.strides = layer.strides
        self.padding = layer.padding
        self.data_format = layer.data_format
        self.dimensions = layer.__class__.__name__[-2]

    def post_pre_train_function(self):
        self.is_pretraining = False

    def set_quantization_bits(self, i_bits, f_bits):
        self.i = ops.convert_to_tensor(i_bits)
        self.f = ops.convert_to_tensor(f_bits)

    def build(self, input_shape):
        super().build(input_shape)
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
                self.hgq.build(input_shape)
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

            self.hgq_gamma = self.hgq_gamma
        else:
            self.quantizer = get_fixed_quantizer(round_mode="RND", overflow_mode=self.overflow)

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0
        loss = (ops.sum(self.hgq_weight.quantizer.i) + ops.sum(self.hgq_weight.quantizer.f)) * self.hgq_gamma
        if self.bias is not None:
            loss += (ops.sum(self.hgq_bias.quantizer.i) + ops.sum(self.hgq_bias.quantizer.f)) * self.hgq_gamma
        return loss

    def quantize_i(self, x):
        if self.use_high_granularity_quantization:
            x = self.hgq(x)
        else:
            x = self.quantizer(x, k=ops.convert_to_tensor(1.0), i=self.i, f=self.f, training=True)
        return x

    def call(self, x):
        x = ops.average_pool(
            x,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return self.quantize_i(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "i": self.i,
                "f": self.f,
                "is_pretraining": self.is_pretraining,
                "overflow": self.overflow,
                "hgq_gamma": self.hgq_gamma,
                "hgq_heterogeneous": self.hgq_heterogeneous,
                "pooling": self.pooling,
            }
        )
        return config


def call_post_round_functions(model, rewind, rounds, r):
    if rewind == "round":
        rewind_weights_functions(model)
    elif rewind == "post-ticket-search" and r == rounds - 1:
        rewind_weights_functions(model)
    else:
        post_round_functions(model)


def _prune_and_quantize_layer(layer, use_bias):
    layer_weights = layer.get_weights()
    layer_weight = ops.cast(layer_weights[0], layer_weights[0].dtype)

    layer_bias = ops.cast(layer_weights[1], layer_weights[1].dtype) if use_bias else None
    weight, bias = layer.prune_and_quantize(layer_weight, layer_bias)
    return weight, bias


def remove_pruning_from_model_tf(model, config):
    x = model.layers[0].output
    for layer in model.layers[1:]:
        if isinstance(layer, CompressedLayerDepthwiseConv2dKeras):
            new_layer = DepthwiseConv2D(
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                depthwise_regularizer=layer.depthwise_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            x = new_layer(x)
            use_bias = layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, CompressedLayerConv2dKeras):
            new_layer = Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                kernel_regularizer=layer.kernel_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            x = new_layer(x)
            use_bias = layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            new_layer = SeparableConv2D(
                filters=layer.pointwise_conv.filters,
                kernel_size=layer.depthwise_conv.kernel_size,
                strides=layer.depthwise_conv.strides,
                padding=layer.depthwise_conv.padding,
                dilation_rate=layer.depthwise_conv.dilation_rate,
                use_bias=layer.pointwise_conv.use_bias,
                depthwise_regularizer=layer.depthwise_conv.depthwise_regularizer,
                pointwise_regularizer=layer.pointwise_conv.kernel_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            x = new_layer(x)
            use_bias = layer.pointwise_conv.use_bias
            depthwise_weight, _ = _prune_and_quantize_layer(layer.depthwise_conv, False)
            pointwise_weight, bias = _prune_and_quantize_layer(layer.pointwise_conv, layer.pointwise_conv.use_bias)
            new_layer.set_weights(
                [depthwise_weight, pointwise_weight, bias] if use_bias else [depthwise_weight, pointwise_weight]
            )

        elif isinstance(layer, CompressedLayerConv1dKeras):
            new_layer = Conv1D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                kernel_regularizer=layer.kernel_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            x = new_layer(x)
            use_bias = layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, CompressedLayerDenseKeras):
            new_layer = Dense(units=layer.units, use_bias=layer.use_bias, kernel_regularizer=layer.kernel_regularizer)
            x = new_layer(x)
            use_bias = new_layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        else:
            x = layer(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    return replaced_model


def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
            layer.pointwise_conv.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.pruning_layer.pre_epoch_function(epoch, total_epochs)
            layer.pointwise_conv.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.pruning_layer.post_round_function()
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.pruning_layer.post_round_function()
            layer.pointwise_conv.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.save_weights()
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.save_weights()
            layer.pointwise_conv.save_weights()


def rewind_weights_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.rewind_weights()
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.rewind_weights()
            layer.pointwise_conv.rewind_weights()


def pre_finetune_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.pruning_layer.pre_finetune_function()
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.pruning_layer.pre_finetune_function()
            layer.pointwise_conv.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            layer.pruning_layer.post_pre_train_function()
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            layer.depthwise_conv.pruning_layer.post_pre_train_function()
            layer.pointwise_conv.pruning_layer.post_pre_train_function()
        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
            layer.post_pre_train_function()
    if config["pruning_parameters"]["pruning_method"] == "pdp" or (
        config["pruning_parameters"]["pruning_method"] == "wanda"
        and config["pruning_parameters"]["calculate_pruning_budget"]
    ):
        pdp_setup(model, config)


def pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = None
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            if global_weights is None:
                global_weights = ops.ravel(layer.weight)
            else:
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.weight)))
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            if global_weights is None:
                global_weights = ops.ravel(layer.depthwise_conv.weight)
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.pointwise_conv.weight)))
            else:
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.depthwise_conv.weight)))
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.pointwise_conv.weight)))

    abs_global_weights = ops.abs(global_weights)
    global_weight_topk, _ = ops.top_k(abs_global_weights, ops.size(abs_global_weights))
    threshold = global_weight_topk[int((1 - config["pruning_parameters"]["sparsity"]) * float(ops.size(global_weight_topk)))]
    global_weights_below_threshold = ops.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            weight_size = ops.size(layer.weight)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = ops.convert_to_tensor(w / weight_size, dtype=layer.weight.dtype)
            layer.pruning_layer.sparsity = ops.convert_to_tensor(w / weight_size, dtype=layer.weight.dtype)  # Wanda
            idx += weight_size
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            weight_size = ops.size(layer.depthwise_conv.weight)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.depthwise_conv.pruning_layer.init_r = ops.convert_to_tensor(
                w / weight_size, dtype=layer.depthwise_conv.weight.dtype
            )
            layer.depthwise_conv.pruning_layer.sparsity = ops.convert_to_tensor(
                w / weight_size, dtype=layer.depthwise_conv.weight.dtype
            )  # Wanda
            idx += weight_size

            weight_size = ops.size(layer.pointwise_conv.weight)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pointwise_conv.pruning_layer.init_r = ops.convert_to_tensor(
                w / weight_size, dtype=layer.pointwise_conv.weight.dtype
            )
            layer.pointwise_conv.pruning_layer.sparsity = ops.convert_to_tensor(
                w / weight_size, dtype=layer.pointwise_conv.weight.dtype
            )  # Wanda
            idx += weight_size


def get_layer_keep_ratio_tf(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            # weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            weight = ops.cast(layer.weight, layer.weight.dtype)
            bias = ops.cast(layer.bias, layer.bias.dtype) if layer.bias is not None else None
            weight, bias = layer.quantize_i(weight, bias)
            transpose = layer.weight_transpose
            if layer.enable_pruning:
                weight = layer.pruning_layer.get_hard_mask(ops.transpose(weight, transpose)) * ops.transpose(
                    weight, transpose
                )
            total_w += ops.size(weight)
            rem = ops.count_nonzero(weight)
            remaining_weights += rem
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            depthwise_weight = ops.cast(layer.depthwise_conv.weight, layer.depthwise_conv.weight.dtype)
            pointwise_weight = ops.cast(layer.pointwise_conv.weight, layer.pointwise_conv.weight.dtype)
            bias = (
                ops.cast(layer.pointwise_conv.bias, layer.pointwise_conv.bias.dtype)
                if layer.pointwise_conv.bias is not None
                else None
            )

            depthwise_weight, _ = layer.depthwise_conv.quantize_i(depthwise_weight, None)
            transpose = layer.depthwise_conv.weight_transpose
            if layer.depthwise_conv.enable_pruning:
                depthwise_weight = layer.depthwise_conv.pruning_layer.get_hard_mask(
                    ops.transpose(depthwise_weight, transpose)
                ) * ops.transpose(depthwise_weight, transpose)
            total_w += ops.size(layer.depthwise_conv.weight)
            rem = ops.count_nonzero(depthwise_weight)
            remaining_weights += rem

            pointwise_weight, _ = layer.pointwise_conv.quantize_i(pointwise_weight, bias)
            transpose = layer.pointwise_conv.weight_transpose
            if layer.pointwise_conv.enable_pruning:
                pointwise_weight = layer.pointwise_conv.pruning_layer.get_hard_mask(
                    ops.transpose(pointwise_weight, transpose)
                ) * ops.transpose(pointwise_weight, transpose)
            total_w += ops.size(layer.pointwise_conv.weight)
            rem = ops.count_nonzero(pointwise_weight)
            remaining_weights += rem

        elif isinstance(layer, (Conv2D, Conv1D, DepthwiseConv2D, Dense)):
            weight = layer.kernel
            total_w += ops.size(weight)
            remaining_weights += ops.count_nonzero(weight)
        elif isinstance(layer, SeparableConv2D):
            depthwise_weight = layer.depthwise_kernel
            pointwise_weight = layer.pointwise_kernel
            total_w += ops.size(depthwise_weight)
            total_w += ops.size(pointwise_weight)
            remaining_weights += ops.count_nonzero(depthwise_weight)
            remaining_weights += ops.count_nonzero(pointwise_weight)
    if total_w != 0:
        return remaining_weights / total_w
    return 0.0


def get_model_losses_tf(model, losses):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                CompressedLayerDepthwiseConv2dKeras,
                CompressedLayerConv2dKeras,
                CompressedLayerConv1dKeras,
                CompressedLayerDenseKeras,
            ),
        ):
            loss = layer.pruning_layer.calculate_additional_loss()
            if layer.enable_quantization and layer.use_high_granularity_quantization:
                loss += layer.hgq_loss()
            losses += loss
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            loss = layer.depthwise_conv.pruning_layer.calculate_additional_loss()
            loss += layer.pointwise_conv.pruning_layer.calculate_additional_loss()
            if layer.enable_quantization and layer.use_high_granularity_quantization:
                loss += layer.depthwise_conv.hgq_loss()
                loss += layer.pointwise_conv.hgq_loss()
            losses += loss
        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
            if layer.use_high_granularity_quantization:
                losses += layer.hgq_loss()
    return losses


def check_activation(layer, config):
    """
    Replaces activations with quantized activations.
    The activation can be a part of another layer such as Conv2D, or an Activation layer
    """
    quantization_enabled = config["quantization_parameters"]["enable_quantization"]
    act = None
    if hasattr(layer.activation, "__name__"):
        if layer.activation.__name__ == "relu":
            i_bits, f_bits = get_quantization_bits_activations(config, layer)
            act = QuantizedReLU(config, i_bits, f_bits) if quantization_enabled else ReLU()
            act.build(layer.input.shape)
        elif layer.activation.__name__ == "tanh":
            i_bits, f_bits = get_quantization_bits_activations(config, layer)
            act = QuantizedTanh(config, i=i_bits, f=f_bits) if quantization_enabled else Activation(activation="tanh")
        else:
            act = None
    return act


def add_compression_layers_tf(model, config, input_shape=None):
    # Pruning algorithms assume channels_first format
    # Creates a new functional model from model, replacing certain layers with compressed / quantized variants
    x = model.layers[0].output
    for layer in model.layers[1:]:
        act = None
        if isinstance(layer, DepthwiseConv2D):
            new_layer = CompressedLayerDepthwiseConv2dKeras(config, layer, layer_type="conv")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)

            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, Conv2D):
            new_layer = CompressedLayerConv2dKeras(config, layer, layer_type="conv")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, SeparableConv2D):
            new_layer = CompressedLayerSeparableConv2dKeras(config, layer)
            dw_i_bits_w, dw_f_bits_w, pw_i_bits_w, pw_f_bits_w, pw_i_bits_b, pw_f_bits_b = (
                get_quantization_bits_weights_biases(config, layer)
            )
            new_layer.depthwise_conv.set_quantization_bits(dw_i_bits_w, dw_f_bits_w, pw_i_bits_b, pw_f_bits_b)
            new_layer.pointwise_conv.set_quantization_bits(pw_i_bits_w, pw_f_bits_w, pw_i_bits_b, pw_f_bits_b)
            enable_pruning_depthwise, enable_pruning_pointwise = get_enable_pruning(layer, config)
            new_layer.depthwise_conv.set_enable_pruning(enable_pruning_depthwise)
            new_layer.pointwise_conv.set_enable_pruning(enable_pruning_pointwise)

            pruning_layer_input = layer.depthwise_kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.depthwise_conv.pruning_layer.build(pruning_layer_input.shape)

            pointwise_pruning_layer_input = layer.pointwise_kernel
            transpose_shape = new_layer.weight_transpose
            pointwise_pruning_layer_input = ops.transpose(pointwise_pruning_layer_input, transpose_shape)
            new_layer.pointwise_conv.pruning_layer.build(pointwise_pruning_layer_input.shape)
            new_layer.depthwise_conv.build(x.shape)
            y = new_layer.depthwise_conv(x).shape
            new_layer.pointwise_conv.build(y)
            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, Conv1D):
            new_layer = CompressedLayerConv1dKeras(config, layer, layer_type="conv")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)

            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, Dense):
            new_layer = CompressedLayerDenseKeras(config, layer, layer_type="linear")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            act = check_activation(layer, config)
        # Activation layers
        elif isinstance(layer, ReLU):
            if config["quantization_parameters"]["enable_quantization"]:
                i_bits = config["quantization_parameters"]["default_integer_bits"]
                f_bits = config["quantization_parameters"]["default_fractional_bits"]
                i_bits, f_bits = get_quantization_bits_activations(config, layer)
                new_layer = QuantizedReLU(config, i_bits, f_bits)
                new_layer.build(layer.input.shape)
                x = new_layer(x)
            else:
                x = layer(x)
        elif isinstance(layer, Activation):
            new_layer = check_activation(layer, config)
            if new_layer is not None:
                x = new_layer(x)
        elif isinstance(layer, (AveragePooling1D, AveragePooling2D, AveragePooling3D)):
            if config["quantization_parameters"]["enable_quantization"]:
                i_bits, f_bits = get_quantization_bits_activations(config, layer)
                new_layer = QuantizedPooling(config, layer)
                new_layer.set_quantization_bits(i_bits, f_bits)
                new_layer.build(layer.output.shape)
                x = new_layer(x)
            else:
                x = layer(x)
        else:
            x = layer(x)
        if act is not None:
            x = act(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    return replaced_model


def get_quantization_bits_activations(config, layer):
    i_bits = config["quantization_parameters"]["default_integer_bits"]
    f_bits = config["quantization_parameters"]["default_fractional_bits"]
    if isinstance(layer, ReLU):
        f_bits += 1  # Unsigned, add 1 bit to default value only
    layer_specific = config["quantization_parameters"]["layer_specific"]
    if layer.name in layer_specific:
        if hasattr(layer, "activation") and layer.activation.__name__ in layer_specific[layer.name]:
            i_bits = layer_specific[layer.name][layer.activation.__name__]["integer_bits"]
            f_bits = layer_specific[layer.name][layer.activation.__name__]["fractional_bits"]
        else:
            i_bits = layer_specific[layer.name]["integer_bits"]
            f_bits = layer_specific[layer.name]["fractional_bits"]
    return i_bits, f_bits


def get_quantization_bits_weights_biases(config, layer):
    layer_specific = config["quantization_parameters"]["layer_specific"]
    if isinstance(layer, SeparableConv2D):
        dw_i_bits_w = pw_i_bits_w = pw_i_bits_b = config["quantization_parameters"]["default_integer_bits"]
        dw_f_bits_w = pw_f_bits_w = pw_f_bits_b = config["quantization_parameters"]["default_fractional_bits"]
        if layer.name in layer_specific:
            if "depthwise" in layer_specific[layer.name]:
                if "weight" in layer_specific[layer.name]["depthwise"]:
                    dw_i_bits_w = layer_specific[layer.name]["depthwise"]["weight"]["integer_bits"]
                    dw_f_bits_w = layer_specific[layer.name]["depthwise"]["weight"]["fractional_bits"]
            if "pointwise" in layer_specific[layer.name]:
                if "weight" in layer_specific[layer.name]["pointwise"]:
                    pw_i_bits_w = layer_specific[layer.name]["pointwise"]["weight"]["integer_bits"]
                    pw_f_bits_w = layer_specific[layer.name]["pointwise"]["weight"]["fractional_bits"]
                if "bias" in layer_specific[layer.name]:
                    pw_i_bits_b = layer_specific[layer.name]["pointwise"]["bias"]["integer_bits"]
                    pw_f_bits_b = layer_specific[layer.name]["pointwise"]["bias"]["fractional_bits"]
        return dw_i_bits_w, dw_f_bits_w, pw_i_bits_w, pw_f_bits_w, pw_i_bits_b, pw_f_bits_b
    else:
        i_bits_w = i_bits_b = config["quantization_parameters"]["default_integer_bits"]
        f_bits_w = f_bits_b = config["quantization_parameters"]["default_fractional_bits"]
        if layer.name in layer_specific:
            if "weight" in layer_specific[layer.name]:
                i_bits_w = layer_specific[layer.name]["weight"]["integer_bits"]
                f_bits_w = layer_specific[layer.name]["weight"]["fractional_bits"]
            if "bias" in layer_specific[layer.name]:
                i_bits_b = layer_specific[layer.name]["bias"]["integer_bits"]
                f_bits_b = layer_specific[layer.name]["bias"]["fractional_bits"]
        return i_bits_w, f_bits_w, i_bits_b, f_bits_b


def get_enable_pruning(layer, config):
    enable_pruning = config["pruning_parameters"]["enable_pruning"]
    if isinstance(layer, SeparableConv2D):
        enable_pruning_depthwise = enable_pruning_pointwise = True
        if layer.name + "_depthwise" in config["pruning_parameters"]["disable_pruning_for_layers"]:
            enable_pruning_depthwise = False
        if layer.name + "pointwise" in config["pruning_parameters"]["disable_pruning_for_layers"]:
            enable_pruning_pointwise = False
        return enable_pruning_depthwise, enable_pruning_pointwise
    else:
        if layer.name in config["pruning_parameters"]["disable_pruning_for_layers"]:
            enable_pruning = False
        return enable_pruning


def add_default_layer_quantization_pruning_to_config_tf(model, config):
    custom_scheme = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for layer in model.layers:
        if layer.__class__ in [Dense, Conv2D, Conv1D, DepthwiseConv2D]:
            if layer.use_bias:
                custom_scheme["layer_specific"][layer.name] = {
                    "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
                }
            else:
                custom_scheme["layer_specific"][layer.name] = {"weight": {"integer_bits": 0.0, "fractional_bits": 7.0}}
            if hasattr(layer.activation, "__name__") and layer.activation.__name__ in ["relu", "tanh"]:
                custom_scheme["layer_specific"][layer.name][layer.activation.__name__] = {
                    "integer_bits": 0.0,
                    "fractional_bits": 7.0,
                }
            custom_scheme["disable_pruning_for_layers"].append(layer.name)
        if layer.__class__ == SeparableConv2D:
            if layer.use_bias:
                custom_scheme["layer_specific"][layer.name] = {
                    "depthwise": {
                        "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    },
                    "pointwise": {
                        "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                        "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    },
                }
            else:
                custom_scheme["layer_specific"][layer.name] = {
                    "depthwise": {"weight": {"integer_bits": 0.0, "fractional_bits": 7.0}},
                    "pointwise": {"weight": {"integer_bits": 0.0, "fractional_bits": 7.0}},
                }
            if hasattr(layer.activation, "__name__") and layer.activation.__name__ in ["relu", "tanh"]:
                custom_scheme["layer_specific"][layer.name][layer.activation.__name__] = {
                    "integer_bits": 0.0,
                    "fractional_bits": 7.0,
                }
            custom_scheme["disable_pruning_for_layers"].append(layer.name + "_depthwise")
            custom_scheme["disable_pruning_for_layers"].append(layer.name + "_pointwise")
        elif layer.__class__ in [Activation, ReLU, AveragePooling1D, AveragePooling2D, AveragePooling3D]:
            custom_scheme["layer_specific"][layer.name] = {"integer_bits": 0.0, "fractional_bits": 7.0}
    config["quantization_parameters"]["layer_specific"] = custom_scheme["layer_specific"]
    config["pruning_parameters"]["disable_pruning_for_layers"] = custom_scheme["disable_pruning_for_layers"]
    return config
