import keras
from hgq.quantizer import Quantizer
from keras import ops
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
        self.quantizer = get_fixed_quantizer(overflow_mode=self.overflow)
        self.weight = self.add_weight(layer.kernel.shape, trainable=True)
        self.init_weight = self.weight.value
        self.hgq_gamma = config["quantization_parameters"]["hgq_gamma"]

        self.bias = self.add_weight(layer.bias.shape, trainable=True) if layer.bias is not None else None
        self.pruning_first = config["training_parameters"]["pruning_first"]
        self.enable_quantization = config["quantization_parameters"]["enable_quantization"]
        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.enable_pruning = config["pruning_parameters"]
        self.channel_format = keras.backend.image_data_format()
        self.transpose = None

        if hasattr(layer, "use_bias"):
            self.use_bias = layer.use_bias
        else:
            self.use_bias = self.bias is not None

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
            self.hgq_weight = Quantizer(
                k0=1.0,
                i0=self.i_weight,
                f0=self.f_weight,
                round_mode="RND",
                overflow_mode=self.overflow,
                q_type="kif",
            )
            self.hgq_weight.build(self.weight.shape)
            if self.use_bias:
                self.hgq_bias = Quantizer(
                    k0=1.0,
                    i0=self.i_bias,
                    f0=self.RND,
                    round_mode="TRN",
                    overflow_mode=self.overflow,
                    q_type="kif",
                )
                self.hgq_bias.build(self.bias.shape)

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
            if self.channel_format == "channels_last":
                weight = ops.transpose(weight, self.transpose)
            weight = self.pruning_layer(weight)
            if self.channel_format == "channels_last":  # Transpose back
                weight = ops.transpose(weight, self.transpose)
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


class CompressedLayerDepthwiseConv2dKeras(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.depthwise_regularizer = layer.depthwise_regularizer
        self.use_bias = layer.use_bias
        self.strides = layer.strides
        self.dilation_rate = layer.dilation_rate
        self.padding = layer.padding
        self.kernel_size = layer.kernel_size
        self.transpose = (2, 3, 0, 1)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, weight, training)
        x = ops.depthwise_conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, training)
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
        self.groups = layer.groups
        self.transpose = (2, 3, 0, 1)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, weight, training)
        x = ops.conv(
            x, weight, strides=self.strides, padding=self.padding, data_format=None, dilation_rate=self.dilation_rate
        )
        if self.bias is not None:
            x = ops.add(x, bias)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, training)
        return x


class CompressedLayerDenseKeras(CompressedLayerBase):
    def __init__(self, config, layer, layer_type):
        super().__init__(config, layer, layer_type)
        self.kernel_regularizer = layer.kernel_regularizer
        self.use_bias = layer.use_bias
        self.units = layer.units
        self.transpose = (1, 0)

    def call(self, x, training=None):
        weight, bias = self.prune_and_quantize(self.weight, self.bias)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, weight, training)
        x = ops.matmul(x, weight)
        if self.bias is not None:
            x = ops.add(x, bias)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, training)
        return x


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
            new_layer = keras.layers.DepthwiseConv2D(
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
            new_layer = keras.layers.Conv2D(
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
            new_layer = keras.layers.Dense(
                units=layer.units, use_bias=layer.use_bias, kernel_regularizer=layer.kernel_regularizer
            )
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
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.save_weights()


def rewind_weights_functions(model):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.rewind_weights()


def pre_finetune_functions(model):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            layer.pruning_layer.post_pre_train_function()
    if config["pruning_parameters"]["pruning_method"] == "pdp" or config["pruning_parameters"]["pruning_method"] == "wanda":
        pdp_setup(model, config)


def pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = None
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            if global_weights is None:
                global_weights = ops.reshape(layer.weight, -1)
            else:
                global_weights = ops.concatenate((global_weights, ops.reshape(layer.weight, -1)))

    abs_global_weights = ops.abs(global_weights)
    global_weight_topk, _ = ops.top_k(abs_global_weights, ops.size(abs_global_weights))
    threshold = global_weight_topk[int((1 - config["pruning_parameters"]["sparsity"]) * float(ops.size(global_weight_topk)))]
    global_weights_below_threshold = ops.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            weight_size = ops.size(layer.weight)
            w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = ops.convert_to_tensor(w / weight_size, dtype=layer.weight.dtype)
            layer.pruning_layer.sparsity = ops.convert_to_tensor(w / weight_size, dtype=layer.weight.dtype)  # Wanda
            idx += weight_size


def get_layer_keep_ratio_tf(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            # weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            weight = ops.cast(layer.weight, layer.weight.dtype)
            bias = ops.cast(layer.bias, layer.bias.dtype) if layer.bias is not None else None
            weight, bias = layer.quantize_i(weight, bias)
            transpose = layer.transpose
            weight = layer.pruning_layer.get_hard_mask(ops.transpose(weight, transpose)) * ops.transpose(weight, transpose)
            total_w += ops.size(layer.weight)
            rem = ops.count_nonzero(weight)
            remaining_weights += rem
        elif isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.Dense)):
            weight = layer.get_weights()[0]
            total_w += ops.size(weight)
            remaining_weights += ops.count_nonzero(weight)
    if total_w != 0:
        return remaining_weights / total_w
    return 0.0


def get_model_losses_tf(model, losses):
    for layer in model.layers:
        if isinstance(layer, (CompressedLayerDepthwiseConv2dKeras, CompressedLayerConv2dKeras, CompressedLayerDenseKeras)):
            loss = layer.pruning_layer.calculate_additional_loss()
            if layer.use_high_granularity_quantization:
                loss += layer.hgq_loss()
            losses += loss
    return losses


def check_activation(layer, config):
    quantization_enabled = config["quantization_parameters"]["enable_quantization"]
    # Replaces activations that are part of a layer, by adding them as a layer
    act = None
    if hasattr(layer.activation, "__name__"):
        if layer.activation.__name__ == "relu":
            i_bits, f_bits = get_quantization_bits_activations(config, layer)
            act = QuantizedReLU(config, i_bits, f_bits) if quantization_enabled else keras.layers.ReLU()
        elif layer.activation.__name__ == "tanh":
            i_bits, f_bits = get_quantization_bits_activations(config, layer)
            act = (
                QuantizedTanh(config, i=i_bits, f=f_bits)
                if quantization_enabled
                else keras.layers.Activation(activation="tanh")
            )
        else:
            act = None
    return act


def add_compression_layers_tf(model, config, input_shape=None):
    requires_transpose = (
        keras.backend.image_data_format() == "channels_last"
    )  # Pruning algorithms assume channels_first format
    # Creates a new functional model from model, replacing certain layers with compressed / quantized variants
    x = model.layers[0].output
    for layer in model.layers[1:]:
        act = None
        if isinstance(layer, (keras.layers.DepthwiseConv2D)):
            new_layer = CompressedLayerDepthwiseConv2dKeras(config, layer, layer_type="conv")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer.name)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = new_layer.weight
            if requires_transpose:
                transpose_shape = (2, 3, 0, 1)
                pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, (keras.layers.Conv2D)):
            new_layer = CompressedLayerConv2dKeras(config, layer, layer_type="conv")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer.name)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = new_layer.weight
            if requires_transpose:
                transpose_shape = (2, 3, 0, 1)
                pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            act = check_activation(layer, config)
        elif isinstance(layer, (keras.layers.Dense)):
            new_layer = CompressedLayerDenseKeras(config, layer, layer_type="linear")
            i_bits_w, f_bits_w, i_bits_b, f_bits_b = get_quantization_bits_weights_biases(config, layer.name)
            new_layer.set_quantization_bits(i_bits_w, f_bits_w, i_bits_b, f_bits_b)
            enable_pruning = get_enable_pruning(layer, config)
            new_layer.set_enable_pruning(enable_pruning)
            pruning_layer_input = new_layer.weight
            if requires_transpose:
                transpose_shape = (1, 0)
                pruning_layer_input = ops.transpose(pruning_layer_input, transpose_shape)
            new_layer.pruning_layer.build(pruning_layer_input.shape)
            x = new_layer(x)
            act = check_activation(layer, config)
        # Activation layers
        elif isinstance(layer, (keras.layers.ReLU)):
            i_bits, f_bits = get_quantization_bits_activations(config, layer)
            new_layer = QuantizedReLU(config, i_bits, f_bits)
            x = new_layer(x)
        elif isinstance(layer, keras.layers.Activation):
            layer = check_activation(layer, config)
            x = layer(x)
        else:
            x = layer(x)
        if act is not None:
            x = act(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    replaced_model(keras.random.normal(shape=input_shape))
    return replaced_model


def get_quantization_bits_activations(config, name):
    i_bits = config["quantization_parameters"]["default_integer_bits"]
    f_bits = config["quantization_parameters"]["default_fractional_bits"]
    layer_specific = config["quantization_parameters"]["layer_specific"]
    if name in layer_specific:
        i_bits = layer_specific[name]["weight"]["integer_bits"]
        f_bits = layer_specific[name]["weight"]["fractional_bits"]
    return i_bits, f_bits


def get_quantization_bits_weights_biases(config, name):
    i_bits_w = i_bits_b = config["quantization_parameters"]["default_integer_bits"]
    f_bits_w = f_bits_b = config["quantization_parameters"]["default_fractional_bits"]
    layer_specific = config["quantization_parameters"]["layer_specific"]
    if name in layer_specific:
        if "weight" in layer_specific[name]:
            i_bits_w = layer_specific[name]["weight"]["integer_bits"]
            f_bits_w = layer_specific[name]["weight"]["fractional_bits"]
        if "bias" in layer_specific[name]:
            i_bits_b = layer_specific[name]["bias"]["integer_bits"]
            f_bits_b = layer_specific[name]["bias"]["fractional_bits"]
    return i_bits_w, f_bits_w, i_bits_b, f_bits_b


def get_enable_pruning(layer, config):
    enable_pruning = config["pruning_parameters"]["enable_pruning"]
    if layer.name in config["pruning_parameters"]["disable_pruning_for_layers"]:
        enable_pruning = False
    return enable_pruning
