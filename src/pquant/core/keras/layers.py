from typing import Tuple, TypeVar

import keras
from keras import ops
from keras.layers import (
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Layer,
    ReLU,
    SeparableConv2D,
)
from keras.src.layers.input_spec import InputSpec
from keras.src.ops.operation_utils import compute_pooling_output_shape

from pquant.core.keras.activations import PQActivation
from pquant.core.keras.quantizer import Quantizer
from pquant.core.utils import get_pruning_layer

T = TypeVar("T")


class PQWeightBiasBase(keras.layers.Layer):
    def __init__(
        self,
        config,
        layer_type,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if weight_quant_bits is not None:
            self.k_weight, self.i_weight, self.f_weight = weight_quant_bits
        else:
            self.k_weight = config.quantization_parameters.default_weight_keep_negatives
            self.i_weight = config.quantization_parameters.default_weight_integer_bits
            self.f_weight = config.quantization_parameters.default_weight_fractional_bits
        if bias_quant_bits is not None:
            self.k_bias, self.i_bias, self.f_bias = bias_quant_bits
        else:
            self.k_bias = config.quantization_parameters.default_weight_keep_negatives
            self.i_bias = config.quantization_parameters.default_weight_integer_bits
            self.f_bias = config.quantization_parameters.default_weight_fractional_bits

        if out_quant_bits is not None:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        else:
            self.k_output = config.quantization_parameters.default_data_keep_negatives
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits

        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config.pruning_parameters.pruning_method
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

        self.pruning_first = config.training_parameters.pruning_first
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_pruning = config.pruning_parameters.enable_pruning
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.final_compression_done = False
        self.built = False
        self.parallelization_factor = -1
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.input_shape = None
        self.is_pretraining = True

    def set_enable_pruning(self, enable_pruning):
        self.enable_pruning = enable_pruning

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def build(self, input_shape):
        super().build(input_shape)
        self.weight_quantizer = Quantizer(
            ops.convert_to_tensor(self.k_weight),
            ops.convert_to_tensor(self.i_weight),
            ops.convert_to_tensor(self.f_weight),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            False,
            self.hgq_gamma,
        )

        # if self.use_bias:
        self.bias_quantizer = Quantizer(
            ops.convert_to_tensor(self.k_bias),
            ops.convert_to_tensor(self.i_bias),
            ops.convert_to_tensor(self.f_bias),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            False,
            self.hgq_gamma,
        )
        self.input_quantizer = Quantizer(
            ops.convert_to_tensor(self.k_input),
            ops.convert_to_tensor(self.i_input),
            ops.convert_to_tensor(self.f_input),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            True,
            self.hgq_gamma,
        )
        self.output_quantizer = Quantizer(
            ops.convert_to_tensor(self.k_output),
            ops.convert_to_tensor(self.i_output),
            ops.convert_to_tensor(self.f_output),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            True,
            self.hgq_gamma,
        )
        self.input_shape = (1,) + input_shape[1:]
        self.n_parallel = ops.prod(input_shape[1:-1])
        self.parallelization_factor = self.parallelization_factor if self.parallelization_factor > 0 else self.n_parallel

    def apply_final_compression(self):
        pass

    def post_pre_train_function(self):
        self.is_pretraining = False
        if self.pruning_layer is not None:
            self.pruning_layer.post_pre_train_function()

    def save_weights(self):
        self.init_weight = self.weight.value

    def rewind_weights(self):
        self.weight.assign(self.init_weight)

    def ebops(self):
        return 0.0

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        if self._bias is not None:
            loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def handle_transpose(self, x, transpose, do_transpose=False):
        if do_transpose:
            x = ops.transpose(x, transpose)
        return x

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.handle_transpose(weight, self.weight_transpose, True)
            weight = self.pruning_layer(weight)
            weight = self.handle_transpose(weight, self.weight_transpose_back, True)
        return weight

    def pre_forward(self, x, training=None):
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x, training=training)
        if self.pruning_method == "wanda":
            self.collect_input(x, self._kernel, training)
        return x

    def post_forward(self, x, training=None):
        if self.quantize_output and self.enable_quantization:
            x = self.output_quantizer(x, training=training)
        if self.pruning_method == "activation_pruning":
            self.collect_output(x, training)
        return x

    def collect_input(self, x, weight, training):
        collect_x = self.handle_transpose(x, self.data_transpose, self.do_transpose_data)
        weight_channels_first = self.handle_transpose(weight, self.weight_transpose, True)
        self.pruning_layer.collect_input(collect_x, weight_channels_first, training)

    def collect_output(self, x, training):
        collect_x = self.handle_transpose(x, self.data_transpose, self.do_transpose_data)
        self.pruning_layer.collect_output(collect_x, training)


class PQDepthwiseConv2d(PQWeightBiasBase, keras.layers.DepthwiseConv2D):
    def __init__(
        self,
        config,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        quantize_input=True,
        quantize_output=False,
        bias: bool = True,
        device=None,
        dtype=None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_regularizer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            **kwargs,
        )
        self.depthwise_regularizer = depthwise_regularizer
        self.use_bias = use_bias
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.weight_transpose = (2, 3, 0, 1)
        self.weight_transpose_back = (2, 3, 0, 1)
        self.data_transpose = (0, 3, 1, 2)
        self.do_transpose_data = self.data_format == "channels_last"
        self._weight = None
        self._bias = None

    def build(self, input_shape):
        super().build(input_shape)
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(min_ndim=self.rank + 2, axes={channel_axis: input_channel})
        depthwise_shape = self.kernel_size + (
            input_channel,
            self.depth_multiplier,
        )
        self._kernel = self.add_weight(
            name="kernel",
            shape=depthwise_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.depth_multiplier * input_channel,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self._bias = None
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.weight_quantizer.build(self._kernel.shape)
            if self.use_bias:
                self.bias_quantizer.build(self._bias.shape)
            self.output_quantizer.build(self.compute_output_shape(input_shape))
        self.input_shape = (1,) + input_shape[1:]

    @property
    def kernel(self):
        if self.final_compression_done:
            return self._kernel
        if self.pruning_first:
            weight = self.prune(self._kernel)
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return weight
        else:
            weight = self._kernel
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return self.prune(weight)

    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    @property
    def bias(self):
        if self.final_compression_done or self._bias is None:
            return self._bias
        bias = self._bias
        if self.enable_quantization:
            bias = self.bias_quantizer(self._bias)
        return bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._kernel))
        if include_mask:
            mask = self.handle_transpose(self.pruning_layer.get_hard_mask(), self.weight_transpose_back, do_transpose=True)
            bw_ker = bw_ker * mask
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = ops.cast((ops.abs(self._kernel) > quantization_step_size), self._kernel.dtype)
            bw_ker = bw_ker * step_size_mask
        if self.parallelization_factor < 0:
            ebops = ops.sum(
                ops.depthwise_conv(
                    bw_inp,
                    bw_ker,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=None,
                    dilation_rate=self.dilation_rate,
                )
            )
        else:
            reduce_axis_kernel = tuple(range(0, 3))
            if self.data_format == "channels_last":  # Is channels last
                reduce_axis_input = reduce_axis_kernel
            else:
                reduce_axis_input = (0,) + tuple(range(2, 4))
            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            reduce_axis_kernel = tuple(range(0, 2))
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)
            ebops = ops.sum(bw_inp[:, None] * bw_ker)
        if self.use_bias:
            size = ops.cast(ops.prod(self.input_shape), self.dtype)
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = super().call(x)
        x = self.post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x

    # Is it supposed to be like this?
    def apply_final_compression(self):
        self._kernel.assign(self.kernel)
        if self._bias is not None:
            self._bias.assign = self.bias
        self.final_compression_done = True

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            f"in_features={self.in_features} "
            f"out_features={self.out_features} "
            f"bias={self._bias is not None} "
            f"quantize_input={self.quantize_input} "
            f"quantize_output={self.quantize_output} "
        )


class PQConv2d(PQWeightBiasBase, keras.layers.Conv2D):
    def __init__(
        self,
        config,
        filters,
        kernel_size,
        quantize_input=True,
        quantize_output=False,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            **kwargs,
        )

        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        self.do_transpose_data = self.data_format == "channels_last"
        self.use_biase = use_bias

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self._bias = None
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.weight_quantizer.build(self._kernel.shape)
            if self.use_bias:
                self.bias_quantizer.build(self._bias.shape)
            self.output_quantizer.build(self.compute_output_shape(input_shape))

    @property
    def kernel(self):
        if self.final_compression_done:
            return self._kernel
        if self.pruning_first:
            weight = self.prune(self._kernel)
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return weight
        else:
            weight = self._kernel
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done or self._bias is None:
            return self._bias
        bias = self._bias
        if self.enable_quantization:
            bias = self.bias_quantizer(self._bias)
        return bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._kernel))
        if include_mask:
            mask = self.handle_transpose(self.pruning_layer.get_hard_mask(), self.weight_transpose_back, do_transpose=True)
            bw_ker = bw_ker * mask
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = ops.cast((ops.abs(self._kernel) > quantization_step_size), self._kernel.dtype)
            bw_ker = bw_ker * step_size_mask
        if self.parallelization_factor < 0:
            ebops = ops.sum(
                ops.conv(
                    bw_inp,
                    bw_ker,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=None,
                    dilation_rate=self.dilation_rate,
                )
            )
        else:
            reduce_axis_kernel = tuple(range(0, 3))
            if self.do_transpose_data:  # Is channels last
                reduce_axis_input = reduce_axis_kernel
            else:
                reduce_axis_input = (0,) + tuple(range(2, 4))
            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            reduce_axis_kernel = tuple(range(0, 2))
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)

            ebops = ops.sum(bw_inp[:, None] * bw_ker)
        if self.use_bias:
            size = ops.cast(ops.prod(self.input_shape), self.dtype)
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = super().call(x)
        x = self.post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x


class PQSeparableConv2d(Layer):
    def __init__(
        self,
        config,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        depth_multiplier=1,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        quantize_input=True,
        quantize_output=False,
        **kwargs,
    ):
        super().__init__()
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        self.depthwise_conv = PQDepthwiseConv2d(
            config,
            kernel_size,
            strides,
            padding,
            depth_multiplier,
            data_format,
            dilation_rate,
            None,
            use_bias=False,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
            depthwise_constraint=depthwise_constraint,
            quantize_input=quantize_input,
            quantize_output=False,
        )

        self.pointwise_conv = PQConv2d(
            config,
            filters=filters,
            kernel_size=1,
            quantize_input=False,
            quantize_output=quantize_output,
            padding="same",
            data_format=data_format,
            groups=1,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
        )
        self.do_transpose_data = data_format == "channels_last"

    def build(self, input_shape):
        super().build(input_shape)

    def apply_final_compression(self):
        self.depthwise_conv.apply_final_compression()
        self.pointwise_conv.apply_final_compression()

    def call(self, x, training=None):
        x = self.depthwise_conv(x, training=training)
        x = self.pointwise_conv(x, training=training)
        return x


class PQConv1d(PQWeightBiasBase, keras.layers.Conv1D):
    def __init__(
        self,
        config,
        filters,
        kernel_size,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_regularizer,
            bias_constraint=bias_constraint,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            **kwargs,
        )

        self.weight_transpose = (2, 1, 0)
        self.weight_transpose_back = (2, 1, 0)
        self.data_transpose = (0, 2, 1)
        self.do_transpose_data = self.data_format == "channels_last"
        self.use_bias = use_bias

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self._bias = None
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.weight_quantizer.build(self._kernel.shape)
            if self.use_bias:
                self.bias_quantizer.build(self._bias.shape)
            self.output_quantizer.build(self.compute_output_shape(input_shape))

    @property
    def kernel(self):
        if self.final_compression_done:
            return self._kernel
        if self.pruning_first:
            weight = self.prune(self._kernel)
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return weight
        else:
            weight = self._kernel
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done or self._bias is None:
            return self._bias
        bias = self._bias
        if self.enable_quantization:
            bias = self.bias_quantizer(self._bias)
        return bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._kernel))
        if include_mask:
            mask = self.handle_transpose(self.pruning_layer.get_hard_mask(), self.weight_transpose_back, do_transpose=True)
            bw_ker = bw_ker * mask
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = ops.cast((ops.abs(self._kernel) > quantization_step_size), self._kernel.dtype)
            bw_ker = bw_ker * step_size_mask
        if self.parallelization_factor < 0:
            ebops = ops.sum(
                ops.conv(
                    bw_inp,
                    bw_ker,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=None,
                    dilation_rate=self.dilation_rate,
                )
            )
        else:
            reduce_axis_kernel = tuple(range(0, 2))
            if self.do_transpose_data:  # Is channels last
                reduce_axis_input = reduce_axis_kernel
            else:
                reduce_axis_input = (0,) + tuple(range(2, 3))
            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            reduce_axis_kernel = tuple(range(0, 1))
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)
            ebops = ops.sum(bw_inp[:, None] * bw_ker)
        if self.use_bias:
            size = ops.cast(ops.prod(self.input_shape), self.dtype)
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = super().call(x)
        x = self.post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x


class PQDense(PQWeightBiasBase, keras.layers.Dense):
    def __init__(
        self,
        config,
        units,
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        lora_alpha=None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            config=config,
            layer_type="linear",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            **kwargs,
        )
        self.weight_transpose = (1, 0)
        self.weight_transpose_back = (1, 0)
        self.data_transpose = (0, 1)  # Always (BATCH_SIZE, OUT_FEATURES)
        self.do_transpose_data = False
        self.use_bias = use_bias

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self._bias = None
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.weight_quantizer.build(self._kernel.shape)
            if self.use_bias:
                self.bias_quantizer.build(self._bias.shape)
            self.output_quantizer.build(self.compute_output_shape(input_shape))

    @property
    def kernel(self):
        if self.final_compression_done:
            return self._kernel
        if self.pruning_first:
            weight = self.prune(self._kernel)
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return weight
        else:
            weight = self._kernel
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done or self._bias is None:
            return self._bias
        bias = self._bias
        if self.enable_quantization:
            bias = self.bias_quantizer(self._bias)
        return bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._kernel))
        if include_mask:
            mask = self.handle_transpose(self.pruning_layer.get_hard_mask(), self.weight_transpose_back, do_transpose=True)
            bw_ker = bw_ker * mask
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = ops.cast((ops.abs(self._kernel) > quantization_step_size), self._kernel.dtype)
            bw_ker = bw_ker * step_size_mask
        ebops = ops.sum(ops.matmul(bw_inp, bw_ker))
        ebops = ebops * self.n_parallel / self.parallelization_factor
        if self.use_bias:
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            size = ops.cast(ops.prod(self.input_shape), self.dtype)
            ebops += ops.mean(bw_bias) * size
        return ebops

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = ops.matmul(x, self.kernel)
        bias = self.bias
        if bias is not None:
            x = ops.add(x, bias)
        x = self.post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x


class PQBatchNormalization(keras.layers.BatchNormalization):
    def __init__(
        self,
        config,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        synchronized=False,
        quantize_input=True,
        **kwargs,
    ):
        super().__init__(
            axis,
            momentum,
            epsilon,
            center,
            scale,
            beta_initializer,
            gamma_initializer,
            moving_mean_initializer,
            moving_variance_initializer,
            beta_regularizer,
            gamma_regularizer,
            beta_constraint,
            gamma_constraint,
            synchronized,
            **kwargs,
        )
        self.overflow = config.quantization_parameters.overflow
        self.round_mode = config.quantization_parameters.round_mode
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.data_k = config.quantization_parameters.default_data_keep_negatives
        self.weight_k = config.quantization_parameters.default_weight_keep_negatives
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.quantize_input = quantize_input
        self.config = config
        self.f_weight = self.f_bias = ops.convert_to_tensor(config.quantization_parameters.default_weight_fractional_bits)
        self.i_weight = self.i_bias = ops.convert_to_tensor(config.quantization_parameters.default_weight_integer_bits)
        self.i_input = ops.convert_to_tensor(config.quantization_parameters.default_data_integer_bits)
        self.f_input = ops.convert_to_tensor(config.quantization_parameters.default_data_fractional_bits)
        self.final_compression_done = False
        self.is_pretraining = True

    def build(self, input_shape):
        super().build(input_shape)
        self.input_quantizer = Quantizer(
            k=1.0,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        self.weight_quantizer = Quantizer(
            k=1.0,
            i=self.i_weight,
            f=self.f_weight,
            round_mode=self.round_mode,
            overflow=self.overflow,
            is_data=False,
            is_heterogeneous=self.use_hgq,
        )
        self.bias_quantizer = Quantizer(
            k=1.0,
            i=self.i_bias,
            f=self.f_bias,
            round_mode=self.round_mode,
            overflow=self.overflow,
            is_data=False,
            is_heterogeneous=self.use_hgq,
        )
        self.input_quantizer.build(input_shape)
        self.weight_quantizer.build(self.moving_variance.shape)
        self.bias_quantizer.build(self.moving_mean.shape)
        shape = [1] * len(input_shape)
        shape[self.axis] = input_shape[self.axis]
        self._shape = tuple(shape)
        self.input_shape = (1,) + input_shape[1:]

    def apply_final_compression(self):
        self.final_compression_done = True
        gamma, beta = self.gamma, self.beta
        if self.enable_quantization:
            if gamma is not None:
                gamma = self.weight_quantizer(gamma)
                self.gamma.assign(gamma)
            if beta is not None:
                beta = self.bias_quantizer(beta)
                self.beta.assign(beta)

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = ops.reshape(self.weight_quantizer.get_total_bits(self.moving_mean.shape), self._shape)
        bw_bias = ops.reshape(self.bias_quantizer.get_total_bits(self.moving_mean.shape), self._shape)
        size = ops.cast(ops.prod(self.input_shape), self.dtype)
        ebops = ops.sum(bw_inp * bw_ker) + ops.mean(bw_bias) * size
        return ebops

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        return loss

    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        compute_dtype = keras.backend.result_type(inputs.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        inputs = ops.cast(inputs, compute_dtype)
        if self.quantize_input and self.enable_quantization:
            inputs = self.input_quantizer(inputs, training=training)
        moving_mean = ops.cast(self.moving_mean, inputs.dtype)
        moving_variance = ops.cast(self.moving_variance, inputs.dtype)

        if training and self.trainable:
            mean, variance = self._moments(inputs, mask)

            self.moving_mean.assign(moving_mean * self.momentum + mean * (1.0 - self.momentum))
            self.moving_variance.assign(moving_variance * self.momentum + variance * (1.0 - self.momentum))
        else:
            mean = moving_mean
            variance = moving_variance

        if self.scale:
            gamma = self.gamma
            if self.enable_quantization and not self.final_compression_done:
                gamma = self.weight_quantizer(self.gamma)
            gamma = ops.cast(gamma, inputs.dtype)
        else:
            gamma = None

        if self.center:
            beta = self.beta
            if self.enable_quantization and not self.final_compression_done:
                beta = self.bias_quantizer(self.beta)
            beta = ops.cast(beta, inputs.dtype)
        else:
            beta = None

        outputs = ops.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )
        self.add_loss(self.hgq_loss())
        return ops.cast(outputs, self.compute_dtype)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def post_pre_train_function(self):
        self.is_pretraining = False


class PQAvgPoolBase(keras.layers.Layer):
    def __init__(
        self,
        config,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if out_quant_bits is not None:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        else:
            self.k_output = config.quantization_parameters.default_data_keep_negatives
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits

        self.overflow = config.quantization_parameters.overflow
        self.config = config
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.data_k = config.quantization_parameters.default_data_keep_negatives
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.saved_inputs = []
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

    def post_pre_train_function(self):
        self.is_pretraining = False

    def build(self, input_shape):
        self.input_quantizer = Quantizer(
            k=1.0,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        self.output_quantizer = Quantizer(
            k=1.0,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.output_quantizer.build(self.compute_output_shape(input_shape))
        self.input_shape = (1,) + input_shape[1:]

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def compute_output_shape(self, input_shape):
        return compute_pooling_output_shape(
            input_shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def pre_pooling(self, x, training):
        if not hasattr(self, "input_quantizer"):
            self.build(x.shape)
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x, training=training)
        return x

    def post_pooling(self, x, training):
        if self.quantize_output and self.enable_quantization:
            x = self.output_quantizer(x, training=training)
        return x

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        return ops.sum(bw_inp)

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "i_input": self.i_input,
                "f_input": self.f_input,
                "i_output": self.i_output,
                "f_output": self.f_output,
                "is_pretraining": self.is_pretraining,
                "overflow": self.overflow,
                "hgq_gamma": self.hgq_gamma,
                "hgq_heterogeneous": self.hgq_heterogeneous,
                "pooling": self.pooling,
            }
        )
        return config


class PQAvgPool1d(PQAvgPoolBase, keras.layers.AveragePooling1D):
    def __init__(
        self,
        config,
        pool_size,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            out_quant_bits=out_quant_bits,
            **kwargs,
        )

    def call(self, x, training=None):
        x = self.pre_pooling(x, training)
        x = super().call(x)
        x = self.post_pooling(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x


class PQAvgPool2d(PQAvgPoolBase, keras.layers.AveragePooling2D):
    def __init__(
        self,
        config,
        pool_size,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            out_quant_bits=out_quant_bits,
        )

    def call(self, x, training=None):
        x = self.pre_pooling(x, training)
        x = super().call(x)
        x = self.post_pooling(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x


def call_post_round_functions(model, rewind, rounds, r):
    if rewind == "round":
        rewind_weights_functions(model)
    elif rewind == "post-ticket-search" and r == rounds - 1:
        rewind_weights_functions(model)
    else:
        post_round_functions(model)


def apply_final_compression(model):
    x = model.layers[0].output
    for layer in model.layers[1:]:
        if isinstance(layer, (PQWeightBiasBase, PQSeparableConv2d, PQBatchNormalization, PQDepthwiseConv2d)):
            layer.apply_final_compression()
            x = layer(x)
        else:
            x = layer(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    return replaced_model


def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
            layer.pointwise_conv.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.pruning_layer.pre_epoch_function(epoch, total_epochs)
            layer.pointwise_conv.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.pruning_layer.post_round_function()
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.pruning_layer.post_round_function()
            layer.pointwise_conv.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.save_weights()
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.save_weights()
            layer.pointwise_conv.save_weights()


def rewind_weights_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.rewind_weights()
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.rewind_weights()
            layer.pointwise_conv.rewind_weights()


def pre_finetune_functions(model):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.pruning_layer.pre_finetune_function()
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.pruning_layer.pre_finetune_function()
            layer.pointwise_conv.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            layer.pruning_layer.post_pre_train_function()
        elif isinstance(layer, PQSeparableConv2d):
            layer.depthwise_conv.pruning_layer.post_pre_train_function()
            layer.pointwise_conv.pruning_layer.post_pre_train_function()
        elif isinstance(layer, (PQActivation, PQAvgPoolBase, PQBatchNormalization)):
            layer.post_pre_train_function()
    if config.pruning_parameters.pruning_method == "pdp" or (
        config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
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
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            if global_weights is None:
                global_weights = ops.ravel(layer.kernel)
            else:
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.kernel)))
        elif isinstance(layer, PQSeparableConv2d):
            if global_weights is None:
                global_weights = ops.ravel(layer.depthwise_conv.kernel)
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.pointwise_conv.kernel)))
            else:
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.depthwise_conv.kernel)))
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.pointwise_conv.kernel)))

    abs_global_weights = ops.abs(global_weights)
    global_weight_topk, _ = ops.top_k(abs_global_weights, ops.size(abs_global_weights))
    threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * float(ops.size(global_weight_topk)))]
    global_weights_below_threshold = ops.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            weight_size = ops.size(layer.kernel)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = ops.convert_to_tensor(w / weight_size, dtype=layer.kernel.dtype)
            layer.pruning_layer.sparsity = ops.convert_to_tensor(w / weight_size, dtype=layer.kernel.dtype)  # Wanda
            idx += weight_size
        elif isinstance(layer, PQSeparableConv2d):
            weight_size = ops.size(layer.depthwise_conv.kernel)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.depthwise_conv.pruning_layer.init_r = ops.convert_to_tensor(
                w / weight_size, dtype=layer.depthwise_conv.kernel.dtype
            )
            layer.depthwise_conv.pruning_layer.sparsity = ops.convert_to_tensor(
                w / weight_size, dtype=layer.depthwise_conv.kernel.dtype
            )  # Wanda
            idx += weight_size

            weight_size = ops.size(layer.pointwise_conv.kernel)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pointwise_conv.pruning_layer.init_r = ops.convert_to_tensor(
                w / weight_size, dtype=layer.pointwise_conv.kernel.dtype
            )
            layer.pointwise_conv.pruning_layer.sparsity = ops.convert_to_tensor(
                w / weight_size, dtype=layer.pointwise_conv.kernel.dtype
            )  # Wanda
            idx += weight_size


def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            if layer.pruning_first:
                weight = ops.transpose(layer.pruning_layer.get_hard_mask(), layer.weight_transpose_back) * layer._kernel
                if layer.enable_quantization:
                    weight = layer.weight_quantizer(weight)
                weight = weight
            else:
                weight = layer._kernel
                if layer.enable_quantization:
                    weight = layer.weight_quantizer(weight)
                weight = ops.transpose(layer.pruning_layer.get_hard_mask(), layer.weight_transpose_back) * weight
                total_w += ops.size(weight)
                rem = ops.count_nonzero(weight)
                remaining_weights += rem
        elif isinstance(layer, PQSeparableConv2d):
            depthwise_weight = ops.cast(layer.depthwise_conv.kernel, layer.depthwise_conv.kernel.dtype)
            pointwise_weight = ops.cast(layer.pointwise_conv.kernel, layer.pointwise_conv.kernel.dtype)

            depthwise_weight = layer.depthwise_conv.kernel
            transpose = layer.depthwise_conv.weight_transpose
            if layer.depthwise_conv.enable_pruning:
                depthwise_weight = layer.depthwise_conv.pruning_layer.get_hard_mask(
                    ops.transpose(depthwise_weight, transpose)
                ) * ops.transpose(depthwise_weight, transpose)
            total_w += ops.size(layer.depthwise_conv.kernel)
            rem = ops.count_nonzero(depthwise_weight)
            remaining_weights += rem

            pointwise_weight = layer.pointwise_conv.kernel
            transpose = layer.pointwise_conv.weight_transpose
            if layer.pointwise_conv.enable_pruning:
                pointwise_weight = layer.pointwise_conv.pruning_layer.get_hard_mask(
                    ops.transpose(pointwise_weight, transpose)
                ) * ops.transpose(pointwise_weight, transpose)
            total_w += ops.size(layer.pointwise_conv.kernel)
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


def get_model_losses(model, losses):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
            ),
        ):
            loss = layer.pruning_layer.calculate_additional_loss()
            if layer.enable_quantization and layer.use_hgq:
                loss += layer.hgq_loss()
            losses += loss
        elif isinstance(layer, PQSeparableConv2d):
            loss = layer.depthwise_conv.pruning_layer.calculate_additional_loss()
            loss += layer.pointwise_conv.pruning_layer.calculate_additional_loss()
            if layer.enable_quantization and layer.use_hgq:
                loss += layer.depthwise_conv.hgq_loss()
                loss += layer.pointwise_conv.hgq_loss()
            losses += loss
        elif isinstance(layer, (PQActivation, PQAvgPoolBase, PQBatchNormalization)):
            if layer.enable_quantization and layer.use_hgq:
                losses += layer.hgq_loss()
    return losses


def check_activation(layer, config):
    """
    Replaces activations with quantized activations.
    The activation can be a part of another layer such as Conv2D, or an Activation layer
    """
    quantization_enabled = config.quantization_parameters.enable_quantization
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    act = None
    if hasattr(layer.activation, "__name__"):
        if layer.activation.__name__ == "relu":
            act = (
                PQActivation(config, "relu", quantize_input=quantize_input, quantize_output=quantize_output)
                if quantization_enabled
                else ReLU()
            )
            if quantization_enabled:
                set_quantization_bits_activations(config, layer, act)
            act.build(layer.input.shape)
        elif layer.activation.__name__ == "tanh":
            type_of_tanh = "tanh" if config.quantization_parameters.use_real_tanh else "hard_tanh"
            act = (
                PQActivation(config, type_of_tanh, quantize_input=quantize_input, quantize_output=quantize_output)
                if quantization_enabled
                else Activation(activation="tanh")
            )
            if quantization_enabled:
                set_quantization_bits_activations(config, layer, act)
                act.build(layer.input.shape)
        else:
            act = None
    return act


def add_compression_layers(model, config, input_shape=None):
    # Pruning algorithms assume channels_first format
    # Creates a new functional model from model, replacing certain layers with compressed / quantized variants
    x = model.layers[0].output
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    for layer in model.layers[1:]:
        act = None
        if isinstance(layer, DepthwiseConv2D):
            new_layer = PQDepthwiseConv2d(
                config,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                depth_multiplier=layer.depth_multiplier,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                activation=layer.activation,
                use_bias=layer.use_bias,
                bias_initializer=layer.bias_initializer,
                depthwise_initializer=layer.depthwise_initializer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                depthwise_constraint=layer.depthwise_constraint,
                bias_constraint=layer.bias_constraint,
                bias=layer.bias,
                dtype=layer.dtype,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            set_quantization_bits_weight_layers(config, layer, new_layer)

            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)

            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, Conv2D):
            new_layer = PQConv2d(
                config=config,
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                groups=layer.groups,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            set_quantization_bits_weight_layers(config, layer, new_layer)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            new_layer.build(x.shape)
            x = new_layer(x)
            new_layer._kernel.assign(layer._kernel)
            if layer.use_bias:
                new_layer._bias.assign(layer.bias)
            act = check_activation(layer, config)
        elif isinstance(layer, SeparableConv2D):
            new_layer = PQSeparableConv2d(
                config,
                layer.filters,
                layer.kernel_size,
                layer.strides,
                layer.padding,
                layer.data_format,
                layer.dilation_rate,
                layer.depth_multiplier,
                layer.use_bias,
                layer.depthwise_initializer,
                layer.pointwise_initializer,
                layer.bias_initializer,
                layer.depthwise_regularizer,
                layer.pointwise_regularizer,
                layer.bias_regularizer,
                layer.depthwise_constraint,
                layer.pointwise_constraint,
                layer.bias_constraint,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            set_quantization_bits_weight_layers(config, layer, new_layer)

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
            new_layer = PQConv1d(
                config=config,
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                groups=layer.groups,
                activation=None,
                use_bias=layer.use_bias,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            set_quantization_bits_weight_layers(config, layer, new_layer)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            new_layer.build(x.shape)
            x = new_layer(x)
            new_layer._kernel.assign(layer._kernel)
            if layer.use_bias:
                new_layer._bias.assign(layer.bias)
            act = check_activation(layer, config)
        elif isinstance(layer, Dense):
            new_layer = PQDense(
                config=config,
                units=layer.units,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            set_quantization_bits_weight_layers(config, layer, new_layer)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            new_layer._kernel.assign(layer._kernel)
            if layer.use_bias:
                new_layer._bias.assign(layer.bias)
            act = check_activation(layer, config)
        # Activation layers
        elif isinstance(layer, ReLU):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQActivation(config, "relu", quantize_input=quantize_input, quantize_output=quantize_output)
                set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(layer.input.shape)
                x = new_layer(x)

            else:
                x = layer(x)
        elif isinstance(layer, Activation):
            new_layer = check_activation(layer, config)

            if new_layer is not None:
                x = new_layer(x)
        elif isinstance(layer, AveragePooling1D):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQAvgPool1d(
                    config=config,
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    data_format=layer.data_format,
                )
                set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(x.shape)
                x = new_layer(x)
        elif isinstance(layer, AveragePooling2D):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQAvgPool2d(
                    config=config,
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    data_format=layer.data_format,
                )
                set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(x.shape)
                x = new_layer(x)
        elif isinstance(layer, (BatchNormalization)):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQBatchNormalization(
                    config,
                    layer.axis,
                    layer.momentum,
                    layer.epsilon,
                    layer.center,
                    layer.scale,
                    layer.beta_initializer,
                    layer.gamma_initializer,
                    layer.moving_mean_initializer,
                    layer.moving_variance_initializer,
                    layer.beta_regularizer,
                    layer.gamma_regularizer,
                    layer.beta_constraint,
                    layer.gamma_constraint,
                    layer.synchronized,
                    quantize_input=True,
                )
                set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(x.shape)
                x = new_layer(x)
            else:
                x = layer(x)
        else:
            x = layer(x)
        if act is not None:
            x = act(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    return replaced_model


def set_quantization_bits_activations(config, layer, new_layer):
    i_input = i_output = i_weight = i_bias = config.quantization_parameters.default_data_integer_bits
    f_input = f_output = f_weight = f_bias = config.quantization_parameters.default_data_fractional_bits
    if isinstance(layer, ReLU):
        f_input += 1
        f_output += 1  # Unsigned, add 1 bit to default value only
    if layer.name in config.quantization_parameters.layer_specific:
        layer_config = config.quantization_parameters.layer_specific[layer.name]
        if hasattr(layer, "activation") and layer.activation.__name__ in layer_config:
            if "input" in layer_config[layer.activation.__name__]:
                if "integer_bits" in layer_config[layer.activation.__name__]["input"]:
                    i_input = layer_config[layer.activation.__name__]["input"]["integer_bits"]
                if "integer_bits" in layer_config[layer.activation.__name__]["input"]:
                    f_input = layer_config[layer.activation.__name__]["input"]["fractional_bits"]
                if "quantize" in layer_config[layer.activation.__name__]["input"]:
                    new_layer.quantize_input = layer_config[layer.activation.__name__]["input"]["quantize"]
            if "output" in layer_config[layer.activation.__name__]:
                if "integer_bits" in layer_config[layer.activation.__name__]["output"]:
                    i_output = layer_config[layer.activation.__name__]["output"]["integer_bits"]
                if "fractional_bits" in layer_config[layer.activation.__name__]["output"]:
                    f_output = layer_config[layer.activation.__name__]["output"]["fractional_bits"]
                if "quantize" in layer_config[layer.activation.__name__]["output"]:
                    new_layer.quantize_output = layer_config[layer.activation.__name__]["output"]["quantize"]
        else:
            if "input" in layer_config:
                if "integer_bits" in layer_config["input"]:
                    i_input = layer_config["input"]["integer_bits"]
                if "fractional_bits" in layer_config["input"]:
                    f_input = layer_config["input"]["fractional_bits"]
                if "quantize" in layer_config["input"]:
                    new_layer.quantize_input = layer_config["input"]["quantize"]
            if "weight" in layer_config:
                if "integer_bits" in layer_config["weight"]:
                    i_weight = layer_config["weight"]["integer_bits"]
                if "fractional_bits" in layer_config["weight"]:
                    f_weight = layer_config["weight"]["fractional_bits"]
            if "bias" in layer_config:
                if "integer_bits" in layer_config["bias"]:
                    i_bias = layer_config["bias"]["integer_bits"]
                if "fractional_bits" in layer_config["bias"]:
                    f_bias = layer_config["bias"]["fractional_bits"]
            if "output" in layer_config:
                if "integer_bits" in layer_config["output"]:
                    i_output = layer_config["output"]["integer_bits"]
                if "fractional_bits" in layer_config["output"]:
                    f_output = layer_config["output"]["fractional_bits"]
                if "quantize" in layer_config["output"]:
                    new_layer.quantize_output = layer_config["output"]["quantize"]
    if isinstance(layer, BatchNormalization):
        new_layer.i_weight = i_weight
        new_layer.f_weight = f_weight
        new_layer.i_bias = i_bias
        new_layer.f_bias = f_bias
    new_layer.i_input = i_input
    new_layer.f_input = f_input
    new_layer.i_output = i_output
    new_layer.f_output = f_output


def set_quantization_bits_weight_layers(config, layer, new_layer):
    layer_specific = config.quantization_parameters.layer_specific
    if isinstance(layer, SeparableConv2D):
        dw_i_bits_w = pw_i_bits_w = pw_i_bits_b = config.quantization_parameters.default_weight_integer_bits
        dw_f_bits_w = pw_f_bits_w = pw_f_bits_b = config.quantization_parameters.default_weight_fractional_bits
        i_input = i_output = config.quantization_parameters.default_data_integer_bits
        f_input = f_output = config.quantization_parameters.default_data_fractional_bits
        if layer.name in layer_specific:
            layer_config = layer_specific[layer.name]
            if "input" in layer_config:
                if "quantize" in layer_config["input"]:
                    new_layer.depthwise_conv.quantize_input = layer_config["input"]["quantize"]
                if "integer_bits" in layer_config["input"]:
                    i_input = layer_config["input"]["integer_bits"]
                if "fractional_bits" in layer_config["input"]:
                    f_input = layer_config["input"]["fractional_bits"]
            if "depthwise" in layer_config:
                if "weight" in layer_config["depthwise"]:
                    dw_i_bits_w = layer_config["depthwise"]["weight"]["integer_bits"]
                    dw_f_bits_w = layer_config["depthwise"]["weight"]["fractional_bits"]
            if "pointwise" in layer_config:
                if "weight" in layer_config["pointwise"]:
                    pw_i_bits_w = layer_config["pointwise"]["weight"]["integer_bits"]
                    pw_f_bits_w = layer_config["pointwise"]["weight"]["fractional_bits"]
                if "bias" in layer_config:
                    pw_i_bits_b = layer_config["pointwise"]["bias"]["integer_bits"]
                    pw_f_bits_b = layer_config["pointwise"]["bias"]["fractional_bits"]
            if "output" in layer_config:
                if "quantize" in layer_config["output"]:
                    new_layer.quantize_output = layer_config["output"]["quantize"]
                if "integer_bits" in layer_config["output"]:
                    i_output = layer_config["output"]["integer_bits"]
                if "fractional_bits" in layer_config["output"]:
                    f_output = layer_config["output"]["fractional_bits"]
        new_layer.depthwise_conv.i_input = i_input
        new_layer.depthwise_conv.f_input = f_input
        new_layer.depthwise_conv.i_weight = dw_i_bits_w
        new_layer.depthwise_conv.f_weight = dw_f_bits_w
        new_layer.pointwise_conv.i_weight = pw_i_bits_w
        new_layer.pointwise_conv.f_weight = pw_f_bits_w
        new_layer.pointwise_conv.i_bias = pw_i_bits_b
        new_layer.pointwise_conv.f_bias = pw_f_bits_b
        new_layer.pointwise_conv.i_output = i_output
        new_layer.pointwise_conv.f_output = f_output
    else:
        i_bits_w = i_bits_b = config.quantization_parameters.default_weight_integer_bits
        f_bits_w = f_bits_b = config.quantization_parameters.default_weight_fractional_bits
        if layer.name in layer_specific:
            layer_config = layer_specific[layer.name]
            if "input" in layer_config:
                if "quantize" in layer_config["input"]:
                    new_layer.quantize_input = layer_config["input"]["quantize"]
                if "integer_bits" in layer_config["input"]:
                    new_layer.i_input = layer_config["input"]["integer_bits"]
                if "fractional_bits" in layer_config["input"]:
                    new_layer.f_input = layer_config["input"]["fractional_bits"]
            if "weight" in layer_config:
                i_bits_w = layer_config["weight"]["integer_bits"]
                f_bits_w = layer_config["weight"]["fractional_bits"]
            if "bias" in layer_config:
                i_bits_b = layer_config["bias"]["integer_bits"]
                f_bits_b = layer_config["bias"]["fractional_bits"]
            if "output" in layer_config:
                if "quantize" in layer_config["output"]:
                    new_layer.quantize_output = layer_config["output"]["quantize"]
                if "integer_bits" in layer_config["output"]:
                    new_layer.i_output = layer_config["output"]["integer_bits"]
                if "fractional_bits" in layer_config["output"]:
                    new_layer.f_output = layer_config["output"]["fractional_bits"]
        new_layer.i_weight = i_bits_w
        new_layer.f_weight = f_bits_w
        new_layer.i_bias = i_bits_b
        new_layer.f_bias = f_bits_b


def get_enable_pruning(layer, config):
    enable_pruning = config.pruning_parameters.enable_pruning
    if isinstance(layer, (SeparableConv2D, PQSeparableConv2d)):
        enable_pruning_depthwise = enable_pruning_pointwise = True
        if layer.name + "_depthwise" in config.pruning_parameters.disable_pruning_for_layers:
            enable_pruning_depthwise = False
        if layer.name + "pointwise" in config.pruning_parameters.disable_pruning_for_layers:
            enable_pruning_pointwise = False
        return enable_pruning_depthwise, enable_pruning_pointwise
    else:
        if layer.name in config.pruning_parameters.disable_pruning_for_layers:
            enable_pruning = False
        return enable_pruning


def populate_config_with_all_layers(model, config):
    """Create a default config, where all the layers are added to the disable_pruning list, and have their
    own default quantization bits in layer_specific. By default input/output quantization is disabled.
    """
    custom_scheme = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for layer in model.layers:
        if isinstance(layer, (Dense, Conv2D, Conv1D, DepthwiseConv2D, PQWeightBiasBase, PQDepthwiseConv2d)):
            if layer.use_bias:
                custom_scheme["layer_specific"][layer.name] = {
                    "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    "input": {"quantize_input": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "output": {"quantize_input": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            else:
                custom_scheme["layer_specific"][layer.name] = {
                    "input": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                    "weight": {"integer_bits": 0, "fractional_bits": 7},
                    "bias": {"integer_bits": 0, "fractional_bits": 7},
                    "output": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                }
            if hasattr(layer.activation, "__name__") and layer.activation.__name__ in ["relu", "tanh"]:
                custom_scheme["layer_specific"][layer.name][layer.activation.__name__] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            custom_scheme["disable_pruning_for_layers"].append(layer.name)
        if isinstance(layer, (SeparableConv2D, PQSeparableConv2d)):
            if layer.use_bias:
                custom_scheme["layer_specific"][layer.name] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "depthwise": {
                        "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    },
                    "pointwise": {
                        "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                        "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    },
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            else:
                custom_scheme["layer_specific"][layer.name] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "depthwise": {
                        "weight": {
                            "integer_bits": 0.0,
                            "fractional_bits": 7.0,
                        }
                    },
                    "pointwise": {"weight": {"integer_bits": 0.0, "fractional_bits": 7.0}},
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            if hasattr(layer.activation, "__name__") and layer.activation.__name__ in ["relu", "tanh"]:
                custom_scheme["layer_specific"][layer.name][layer.activation.__name__] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            custom_scheme["disable_pruning_for_layers"].append(layer.name + "_depthwise")
            custom_scheme["disable_pruning_for_layers"].append(layer.name + "_pointwise")
        elif isinstance(
            layer, (Activation, ReLU, AveragePooling1D, AveragePooling2D, AveragePooling3D, PQActivation, PQAvgPoolBase)
        ):
            custom_scheme.layer_specific[layer.name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
            }
        elif isinstance(layer, (BatchNormalization, PQBatchNormalization)):
            custom_scheme["layer_specific"][layer.name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
            }
    config.quantization_parameters.layer_specific = custom_scheme["layer_specific"]
    config.pruning_parameters.disable_pruning_for_layers = custom_scheme["disable_pruning_for_layers"]
    return config


def post_training_prune(model, config, calibration_data):
    t_delta = config.pruning_parameters.t_delta
    config.pruning_parameters.t_start_collecting_batch = 0

    for i in range(t_delta):
        inputs = calibration_data[i]
        if i == 0:
            model = add_compression_layers(model, config, inputs.shape)
            post_pretrain_functions(model, config)
        model(inputs, training=True)  # True so pruning works
    return apply_final_compression(model, config)


def get_ebops(model):
    ebops = 0
    for m in model.layers:
        if isinstance(m, (PQWeightBiasBase)):
            ebops += m.ebops(include_mask=m.enable_pruning)
        elif isinstance(m, (PQAvgPoolBase, PQBatchNormalization, PQActivation)):
            ebops += m.ebops()
    return ebops
