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

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.quantizer_functions import create_quantizer
from pquant.core.utils import get_pruning_layer


class PQWeightBiasBase(keras.layers.Layer):
    def __init__(self, config, layer_type, quantize_input=True, quantize_output=False):
        super().__init__()
        i_bits = config.quantization_parameters.default_integer_bits
        f_bits = config.quantization_parameters.default_fractional_bits
        self.data_k = config.quantization_parameters.default_data_keep_negatives
        self.weight_k = config.quantization_parameters.default_weight_keep_negatives
        self.i_weight = ops.convert_to_tensor(i_bits)
        self.f_weight = ops.convert_to_tensor(f_bits)
        self.i_bias = ops.convert_to_tensor(i_bits)
        self.f_bias = ops.convert_to_tensor(f_bits)

        self.i_input = self.i_output = ops.convert_to_tensor(i_bits)
        self.f_input = self.f_output = ops.convert_to_tensor(f_bits)
        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)

        self.pruning_method = config.pruning_parameters.pruning_method
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.pruning_first = config.training_parameters.pruning_first
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.enable_pruning = config.pruning_parameters.enable_pruning
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.final_compression_done = False
        self.do_transpose_data = None
        self.weight_transpose = None
        self.data_transpose = None
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

    def set_input_output_quantization(self, input_quantization, output_quantization):
        self.quantize_input = input_quantization
        self.quantize_output = output_quantization

    def set_input_output_quantization_bits(self, i_input, f_input, i_output, f_output):
        self.i_input = i_input
        self.f_input = f_input
        self.i_output = i_output
        self.f_output = f_output

    def set_enable_pruning(self, enable_pruning):
        self.enable_pruning = enable_pruning

    def build(self, input_shape):
        super().build(input_shape)
        self.weight_quantizer = create_quantizer(
            k=ops.convert_to_tensor(self.weight_k),
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
        )
        if self.use_bias:
            self.bias_quantizer = create_quantizer(
                k=ops.convert_to_tensor(self.weight_k),
                i=self.i_bias,
                f=self.f_bias,
                overflow=self.overflow,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=False,
            )
        if self.quantize_input:
            self.input_quantizer = create_quantizer(
                k=self.data_k,
                i=self.i_input,
                f=self.f_input,
                overflow=self.overflow,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
            )
        if self.quantize_output:
            self.output_quantizer = create_quantizer(
                k=self.data_k,
                i=self.i_output,
                f=self.f_output,
                overflow=self.overflow,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
            )

    def apply_final_compression(self):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        self.weight.assign(weight)
        if self.bias is not None:
            self.bias.assign(bias)
        self.final_compression_done = True

    def save_weights(self):
        self.init_weight = self.weight.value

    def rewind_weights(self):
        self.weight.assign(self.init_weight)

    def ebops(self):
        return 0.0

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining or not self.use_hgq:
            return 0.0
        loss = self.ebops()
        loss += (
            ops.sum(self.weight_quantizer.quantizer.quantizer.i) + ops.sum(self.weight_quantizer.quantizer.quantizer.f)
        ) * self.hgq_gamma
        if self.bias is not None:
            loss += (
                ops.sum(self.bias_quantizer.quantizer.quantizer.i) + ops.sum(self.bias_quantizer.quantizer.quantizer.f)
            ) * self.hgq_gamma
        if self.quantize_input:
            loss += (
                ops.sum(self.input_quantizer.quantizer.quantizer.i) + ops.sum(self.input_quantizer.quantizer.quantizer.f)
            ) * self.hgq_gamma
        if self.quantize_output:
            loss += (
                ops.sum(self.output_quantizer.quantizer.quantizer.i) + ops.sum(self.output_quantizer.quantizer.quantizer.f)
            ) * self.hgq_gamma
        return loss

    def handle_transpose(self, x, transpose, do_transpose=False):
        if do_transpose:
            x = ops.transpose(x, transpose)
        return x

    def quantize_i(self, weight, bias):
        if self.enable_quantization:
            if self.use_hgq:
                weight = self.weight_quantizer(weight)
                bias = None if bias is None else self.bias_quantizer(bias)
            else:
                weight = self.weight_quantizer(
                    weight, k=ops.convert_to_tensor(1.0), i=self.i_weight, f=self.f_weight, training=True
                )
                bias = (
                    None
                    if bias is None
                    else self.bias_quantizer(bias, k=ops.convert_to_tensor(1.0), i=self.i_bias, f=self.f_bias, training=True)
                )
        return weight, bias

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.handle_transpose(weight, self.weight_transpose, True)
            weight = self.pruning_layer(weight)
            weight = self.handle_transpose(weight, self.weight_transpose_back, True)
        return weight

    def prune_and_quantize(self, weight, bias):
        if self.final_compression_done:
            return weight, bias
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

    def pre_forward(self, weight, bias, x, training=None):
        if self.quantize_input:
            if self.use_hgq and not self.input_quantizer.quantizer.built:
                self.input_quantizer.quantizer.build(x.shape)
            if not self.pruning_layer.is_pretraining and not self.use_fitcompress:
                if self.use_hgq:
                    x = self.input_quantizer(x)
                else:
                    x = self.input_quantizer(x, k=self.data_k, i=self.i_input, f=self.f_input)
        if self.pruning_method == "wanda":
            self.collect_input(x, self.weight, training)
        weight, bias = self.prune_and_quantize(weight, bias)
        return weight, bias, x

    def post_forward(self, x, training=None):
        if self.quantize_output:
            if self.use_hgq and not self.output_quantizer.quantizer.built:
                self.output_quantizer.quantizer.build(x.shape)
            if not self.pruning_layer.is_pretraining and not self.use_fitcompress:
                if self.use_hgq:
                    x = self.output_quantizer(x)
                else:
                    x = self.output_quantizer(x, k=self.data_k, i=self.i_output, f=self.f_output)
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


class PQDepthwiseConv2d(PQWeightBiasBase):
    def __init__(self, config, layer, layer_type, quantize_input=True, quantize_output=True):
        super().__init__(config, layer_type, quantize_input, quantize_output)
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
        weight, bias, x = self.pre_forward(self.weight, self.bias, x, training)
        x = ops.depthwise_conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        x = self.post_forward(x, training)
        return x


class PQConv2d(PQWeightBiasBase):
    def __init__(self, config, layer, layer_type, quantize_input=True, quantize_output=False):
        super().__init__(config, layer_type, quantize_input, quantize_output)
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
        weight, bias, x = self.pre_forward(self.weight, self.bias, x, training)
        x = ops.conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.bias is not None:
            x = ops.add(x, bias)
        x = self.post_forward(x, training)
        return x


class PQSeparableConv2d(Layer):
    def __init__(self, config, layer, quantize_input=True, quantize_output=True):
        super().__init__()
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        layer.kernel = layer.depthwise_kernel
        bias = layer.use_bias
        layer.use_bias = False
        self.depthwise_conv = PQDepthwiseConv2d(config, layer, "conv", quantize_input, False)
        layer.kernel_regularizer = layer.pointwise_regularizer
        layer.kernel_size = 1
        layer.kernel = layer.pointwise_kernel
        layer.use_bias = bias
        self.pointwise_conv = PQConv2d(config, layer, "conv", False, quantize_output)
        self.do_transpose_data = layer.data_format == "channels_last"

    def build(self, input_shape):
        super().build(input_shape)

    def apply_final_compression(self):
        self.depthwise_conv.apply_final_compression()
        self.pointwise_conv.apply_final_compression()

    def call(self, x, training=None):
        x = self.depthwise_conv(x, training=training)
        x = self.pointwise_conv(x, training=training)
        return x


class PQConv1d(PQWeightBiasBase):
    def __init__(self, config, layer, layer_type, quantize_input=True, quantize_output=False):
        super().__init__(config, layer_type, quantize_input, quantize_output)
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
        weight, bias, x = self.pre_forward(self.weight, self.bias, x, training)
        x = ops.conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.bias is not None:
            x = ops.add(x, bias)
        x = self.post_forward(x, training)
        return x


class PQDense(PQWeightBiasBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer_type)
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
        self.parallelization_factor = -1

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
        self.input_shape = input_shape
        self.n_parallel = ops.prod(input_shape[1:-1])
        self.parallelization_factor = self.parallelization_factor if self.parallelization_factor > 0 else self.n_parallel

    def ebops(self, shape):
        bw_inp = self.input_quantizer.bits_(shape)
        bw_ker = self.weight_quantizer.bits_(ops.shape(self.weight))
        ebops = ops.sum(ops.matmul(bw_inp, bw_ker))
        ebops = ebops * self.n_parallel / self.parallelization_factor
        if self.use_bias:
            bw_bias = self.bias_quantizer.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(self.input_shape), self.dtype)
            ebops += ops.mean(bw_bias) * size
        return ebops

    def call(self, x, training=None):
        weight, bias, x = self.pre_forward(self.weight, self.bias, x, training)
        x = ops.matmul(x, weight)
        if self.bias is not None:
            x = ops.add(x, bias)
        x = self.post_forward(x, training)
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
        self.overflow = config["quantization_parameters"]["overflow"]
        self.round_mode = config["quantization_parameters"]["round_mode"]
        self.hgq_gamma = config["quantization_parameters"]["hgq_gamma"]
        self.data_k = config["quantization_parameters"]["default_data_keep_negatives"]
        self.weight_k = config["quantization_parameters"]["default_weight_keep_negatives"]
        self.enable_quantization = config["quantization_parameters"]["enable_quantization"]
        self.use_hgq = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.quantize_input = quantize_input
        self.config = config
        self.f_input = self.f_weight = ops.convert_to_tensor(config["quantization_parameters"]["default_fractional_bits"])
        self.i_input = self.i_weight = ops.convert_to_tensor(config["quantization_parameters"]["default_integer_bits"])
        self.final_compression_done = False

    def build(self, input_shape):
        super().build(input_shape)
        self.parameter_quantizer = create_quantizer(
            k=self.weight_k,
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
        )
        self.input_quantizer = create_quantizer(
            k=self.data_k,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
        )

    def apply_final_compression(self):
        gamma, beta = self.gamma, self.beta
        if self.enable_quantization:
            if self.use_hgq:
                gamma = self.parameter_quantizer(gamma)
                beta = self.parameter_quantizer(beta)
            else:
                gamma = self.parameter_quantizer(self.gamma, k=self.data_k, i=self.i_weight, f=self.f_weight)
                beta = self.parameter_quantizer(self.beta, k=self.data_k, i=self.i_weight, f=self.f_weight)
        self.gamma.assign(gamma)
        self.beta.assign(beta)
        self.final_compression_done = True

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
            if self.use_hgq:
                inputs = self.input_quantizer(inputs)
            else:
                inputs = self.input_quantizer(inputs, k=self.data_k, i=self.i_input, f=self.f_input)

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
            if self.enable_quantization and not self.final_compression_done:
                if self.use_hgq:
                    gamma = self.parameter_quantizer(self.gamma)
                else:
                    gamma = self.parameter_quantizer(self.gamma, k=self.weight_k, i=self.i_weight, f=self.f_weight)
            gamma = ops.cast(gamma, inputs.dtype)
        else:
            gamma = None

        if self.center:
            if self.enable_quantization and not self.final_compression_done:
                if self.use_hgq:
                    beta = self.parameter_quantizer(self.beta)
                else:
                    beta = self.parameter_quantizer(self.beta, k=self.weight_k, i=self.i_weight, f=self.f_weight)
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
        return ops.cast(outputs, self.compute_dtype)


class QuantizedPooling(keras.layers.Layer):
    def __init__(self, config, layer, quantize_input=True):
        super().__init__()
        self.i = ops.convert_to_tensor(config.quantization_parameters.default_integer_bits)
        self.f = ops.convert_to_tensor(config.quantization_parameters.default_fractional_bits)

        self.is_pretraining = True

        self.overflow = "SAT_SYM" if config.quantization_parameters.use_symmetric_quantization else "SAT"
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.data_k = config.quantization_parameters.default_data_keep_negatives
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_heterogeneous = config.hgq_heterogeneous
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.pool_size = layer.pool_size
        self.strides = layer.strides
        self.padding = layer.padding
        self.data_format = layer.data_format
        self.quantize_input = quantize_input
        self.dimensions = layer.__class__.__name__[-2]

    def post_pre_train_function(self):
        self.is_pretraining = False

    def build(self, input_shape):
        super().build(input_shape)
        self.input_quantizer = create_quantizer(
            k=self.data_k,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
        )
        self.hgq_gamma = self.hgq_gamma

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        return (ops.sum(self.input_quantizer.quantizer.i) + ops.sum(self.input_quantizer.quantizer.f)) * self.hgq_gamma

    def call(self, x):
        if self.quantize_input and self.enable_quantization:
            if self.use_hgq:
                x = self.input_quantizer(x)
            else:
                x = self.input_quantizer(x, k=self.data_k, i=self.i_input, f=self.f_input)
        return ops.average_pool(
            x,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

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


def apply_final_compression_tf(model):
    x = model.layers[0].output
    for layer in model.layers[1:]:
        if isinstance(layer, (PQWeightBiasBase, PQSeparableConv2d, PQBatchNormalization)):
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
        elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
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
                global_weights = ops.ravel(layer.weight)
            else:
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.weight)))
        elif isinstance(layer, PQSeparableConv2d):
            if global_weights is None:
                global_weights = ops.ravel(layer.depthwise_conv.weight)
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.pointwise_conv.weight)))
            else:
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.depthwise_conv.weight)))
                global_weights = ops.concatenate((global_weights, ops.ravel(layer.pointwise_conv.weight)))

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
            weight_size = ops.size(layer.weight)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = ops.convert_to_tensor(w / weight_size, dtype=layer.weight.dtype)
            layer.pruning_layer.sparsity = ops.convert_to_tensor(w / weight_size, dtype=layer.weight.dtype)  # Wanda
            idx += weight_size
        elif isinstance(layer, PQSeparableConv2d):
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
                PQDepthwiseConv2d,
                PQConv2d,
                PQConv1d,
                PQDense,
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
        elif isinstance(layer, PQSeparableConv2d):
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
    quantization_enabled = config.quantization_parameters.enable_quantization
    act = None
    if hasattr(layer.activation, "__name__"):
        if layer.activation.__name__ == "relu":

            act = QuantizedReLU(config) if quantization_enabled else ReLU()
            if quantization_enabled:
                get_quantization_bits_activations(config, layer, act)
            act.build(layer.input.shape)
        elif layer.activation.__name__ == "tanh":
            act = QuantizedTanh(config) if quantization_enabled else Activation(activation="tanh")
            if quantization_enabled:
                get_quantization_bits_activations(config, layer, act)
                act.build(layer.input.shape)
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
            new_layer = PQDepthwiseConv2d(config, layer, layer_type="conv")
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
            new_layer = PQConv2d(config, layer, layer_type="conv")
            set_quantization_bits_weight_layers(config, layer, new_layer)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, SeparableConv2D):
            new_layer = PQSeparableConv2d(config, layer)
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
            new_layer = PQConv1d(config, layer, layer_type="conv")
            set_quantization_bits_weight_layers(config, layer, new_layer)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = layer.kernel
            transpose_shape = new_layer.weight_transpose
            pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)

            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, Dense):
            new_layer = PQDense(config, layer, layer_type="linear")
            set_quantization_bits_weight_layers(config, layer, new_layer)
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
                new_layer = QuantizedReLU(config)
                get_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(layer.input.shape)
                x = new_layer(x)

            else:
                x = layer(x)
        elif isinstance(layer, Activation):
            new_layer = check_activation(layer, config)

            if new_layer is not None:
                x = new_layer(x)
        elif isinstance(layer, (AveragePooling1D, AveragePooling2D, AveragePooling3D)):
            if config.quantization_parameters.enable_quantization:
                new_layer = QuantizedPooling(config, layer)
                get_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(layer.output.shape)
                x = new_layer(x)

        elif isinstance(layer, (BatchNormalization)):
            if config["quantization_parameters"]["enable_quantization"]:
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
                get_quantization_bits_activations(config, layer, new_layer)
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


def get_quantization_bits_activations(config, layer, new_layer):
    i_input = i_output = config.quantization_parameters.default_integer_bits
    f_input = f_output = config.quantization_parameters.default_fractional_bits
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
            if "output" in layer_config:
                if "integer_bits" in layer_config["output"]:
                    i_output = layer_config["output"]["integer_bits"]
                if "fractional_bits" in layer_config["output"]:
                    f_output = layer_config["output"]["fractional_bits"]
                if "quantize" in layer_config["output"]:
                    new_layer.quantize_input = layer_config["output"]["quantize"]
    new_layer.i_input = i_input
    new_layer.f_input = f_input
    new_layer.i_output = i_output
    new_layer.f_output = f_output


def set_quantization_bits_weight_layers(config, layer, new_layer):
    layer_specific = config["quantization_parameters"]["layer_specific"]
    if isinstance(layer, SeparableConv2D):
        dw_i_bits_w = pw_i_bits_w = pw_i_bits_b = i_input = i_output = config.quantization_parameters.default_integer_bits
        dw_f_bits_w = pw_f_bits_w = pw_f_bits_b = f_input = f_output = config.quantization_parameters.default_fractional_bits
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
                    new_layer.quantize_input = layer_config["output"]["quantize"]
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
        i_bits_w = i_bits_b = config.quantization_parameters.default_integer_bits
        f_bits_w = f_bits_b = config.quantization_parameters.default_fractional_bits
        if layer.name in layer_specific:
            layer_config = layer_specific[layer.name]
            if "input" in layer_config:
                if "quantize" in layer_config["input"]:
                    new_layer.quantize_input = layer_config["input"]["quantize"]
            if "weight" in layer_config:
                i_bits_w = layer_config["weight"]["integer_bits"]
                f_bits_w = layer_config["weight"]["fractional_bits"]
            if "bias" in layer_config:
                i_bits_b = layer_config["bias"]["integer_bits"]
                f_bits_b = layer_config["bias"]["fractional_bits"]
            if "output" in layer_config:
                if "quantize" in layer_config["output"]:
                    new_layer.quantize_output = layer_config["output"]["quantize"]
        new_layer.i_weight = i_bits_w
        new_layer.f_weight = f_bits_w
        new_layer.i_bias = i_bits_b
        new_layer.f_bias = f_bits_b


def get_enable_pruning(layer, config):
    enable_pruning = config.pruning_parameters.enable_pruning
    if isinstance(layer, SeparableConv2D):
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


def add_default_layer_quantization_pruning_to_config_tf(model, config):
    """Create a default config, where all the layers are added to the disable_pruning list, and have their
    own default quantization bits in layer_specific. By default input/output quantization is disabled.
    """
    custom_scheme = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for layer in model.layers:
        if layer.__class__ in [Dense, Conv2D, Conv1D, DepthwiseConv2D]:
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
        if layer.__class__ == SeparableConv2D:
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
        elif layer.__class__ in [Activation, ReLU, AveragePooling1D, AveragePooling2D, AveragePooling3D]:
            custom_scheme.layer_specific[layer.name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
            }
        elif layer.__class__ == BatchNormalization:
            custom_scheme["layer_specific"][layer.name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
            }
    config.quantization_parameters.layer_specific = custom_scheme["layer_specific"]
    config.pruning_parameters.disable_pruning_for_layers = custom_scheme["disable_pruning_for_layers"]
    return config
