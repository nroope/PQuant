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
        self.quantizer = get_fixed_quantizer(overflow_mode=self.overflow)
        self.use_real_tanh = config["quantization_parameters"]["use_real_tanh"]
        if self.use_high_granularity_quantization:
            self.hgq = Quantizer(
                k0=self.k,
                i0=self.i,
                f0=self.f,
                round_mode="RND",
                overflow_mode=self.overflow,
                q_type="kif",
                heterogeneous_axis=(),
            )

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_high_granularity_quantization:
            self.hgq.build(input_shape)

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
        self.quantizer = get_fixed_quantizer(overflow_mode=self.overflow)

        if self.use_high_granularity_quantization:
            self.hgq = Quantizer(
                k0=self.k, i0=self.i, f0=self.f, round_mode="RND", overflow_mode="SAT", q_type="kif", heterogeneous_axis=()
            )

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_high_granularity_quantization:
            self.hgq.build(input_shape)

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
            x = self.quantizer(x, k=0.0, i=self.i, f=self.f, training=True)
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
