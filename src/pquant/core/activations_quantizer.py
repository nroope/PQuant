import keras
from hgq.quantizer import Quantizer
from keras import ops
from keras.ops import convert_to_tensor, maximum, minimum, tanh
from quantizers import get_fixed_quantizer


class QuantizedTanh(keras.layers.Layer):
    def __init__(self, config, i, f):
        super().__init__()
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)
        self.k = convert_to_tensor(1.0)
        self.config = config
        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.is_pretraining = True
        self.overflow = "SAT_SYM" if config["quantization_parameters"]["use_symmetric_quantization"] else "SAT"
        self.use_real_tanh = config["quantization_parameters"]["use_real_tanh"]
        self.hgq_heterogeneous = config["quantization_parameters"]["hgq_heterogeneous"]

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_high_granularity_quantization:
            if self.hgq_heterogeneous:
                self.hgq = Quantizer(
                    k0=self.k,
                    i0=self.i,
                    f0=self.f,
                    round_mode="RND",
                    overflow_mode=self.overflow,
                    q_type="kif",
                    homogeneous_axis=(0,),
                )
            else:
                self.hgq = Quantizer(
                    k0=self.k,
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
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0
        return (ops.sum(self.hgq.quantizer.i) + ops.sum(self.hgq.quantizer.f)) * self.config["quantization_parameters"][
            "hgq_gamma"
        ]

    def post_pre_train_function(self):
        self.is_pretraining = False

    def call(self, x):
        if self.use_high_granularity_quantization:
            x = tanh(x) if self.use_real_tanh else hard_tanh(x)
            return self.hgq(x)
        else:
            x = tanh(x) if self.use_real_tanh else hard_tanh(x)
            x = self.quantizer(x, k=1.0, i=convert_to_tensor(0.0), f=self.f, training=True)
            return x


class QuantizedReLU(keras.layers.Layer):
    def __init__(self, config, i, f):
        super().__init__()
        self.config = config
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)
        self.k = convert_to_tensor(0.0)
        self.use_high_granularity_quantization = config["quantization_parameters"]["use_high_granularity_quantization"]
        self.is_pretraining = True
        self.overflow = "SAT"
        self.use_multiplier = config["quantization_parameters"]["use_relu_multiplier"]
        self.hgq_heterogeneous = config["quantization_parameters"]["hgq_heterogeneous"]
        self.use_fitcompress = config["fitcompress_parameters"]["enable_fitcompress"]
        self.post_fitcompress_calibration = False
        self.saved_inputs = []

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_high_granularity_quantization:
            if self.hgq_heterogeneous:
                self.hgq = Quantizer(
                    k0=self.k,
                    i0=self.i,
                    f0=self.f,
                    round_mode="RND",
                    overflow_mode=self.overflow,
                    q_type="kif",
                    homogeneous_axis=(0,),
                )
            else:
                self.hgq = Quantizer(
                    k0=self.k,
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
        if self.use_multiplier:
            self.multiplier = self.add_weight(shape=(1,), trainable=True, initializer=keras.initializers.Constant(-1.0))

    def set_activation_bits(self, i, f):
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0
        return (ops.sum(self.hgq.quantizer.i) + ops.sum(self.hgq.quantizer.f)) * self.config["quantization_parameters"][
            "hgq_gamma"
        ]

    def call(self, x):
        if self.use_high_granularity_quantization:
            return self.hgq(x)
        else:
            if self.use_fitcompress and self.is_pretraining:
                if self.post_fitcompress_calibration:
                    # Save quantized input into ReLU
                    self.saved_inputs.append(x)
                # During FITcompress, we do not use any quantized activations
                return ops.relu(x)
            # Multiplier after fitcompress if condition, such that we don't use any relu multiplier during FITcompress search
            if self.use_multiplier:
                x = x * 2 ** (ops.stop_gradient(ops.round(self.multiplier) - self.multiplier) + self.multiplier)
            x = self.quantizer(x, k=convert_to_tensor(0.0), i=convert_to_tensor(self.i), f=convert_to_tensor(self.f), training=True)
            return x


def hard_sigmoid(x):
    """Computes hard_sigmoid function that saturates between 0 and 1."""
    x = 0.5 * x + 0.5
    x = maximum(x, 0.0)
    x = minimum(x, 1.0)
    return x


def hard_tanh(x):
    """Computes hard_tanh function that saturates between -1 and 1."""
    return 2.0 * hard_sigmoid(x) - 1.0