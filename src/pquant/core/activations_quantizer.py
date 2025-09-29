import keras
from keras import ops
from keras.ops import convert_to_tensor, maximum, minimum, tanh

from pquant.core.quantizer_functions import create_quantizer


@keras.saving.register_keras_serializable(package="PQuant")
class QuantizedTanh(keras.layers.Layer):
    def __init__(self, config, i, f, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)
        self.k = convert_to_tensor(1.0)
        self.config = config

        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.use_real_tanh = config.quantization_parameters.use_real_tanh
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous

    def build(self, input_shape):
        super().build(input_shape)
        self.quantizer = create_quantizer(
            k=self.k,
            i=self.i,
            f=self.f,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        self.input_quantizer = create_quantizer(
            k=self.k,
            i=self.i,
            f=self.f,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        if self.use_hgq:
            self.quantizer.build(input_shape)

    def set_bits(self, i, f):
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0
        return (
            ops.sum(self.hgq.quantizer.i) + ops.sum(self.hgq.quantizer.f)
        ) * self.config.quantization_parameters.hgq_gamma

    def post_pre_train_function(self):
        self.is_pretraining = False

    def call(self, x):
        if self.use_hgq:
            x = self.input_quantizer(x)
        else:
            x = self.input_quantizer(x, k=self.k, i=self.i, f=self.f)
        x = tanh(x) if self.use_real_tanh else hard_tanh(x)
        if self.use_hgq:
            return self.quantizer(x)
        return self.quantizer(x, k=self.k, i=self.i, f=self.f)

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.get_dict(), "i": float(self.i), "f": float(self.f)})
        return config


@keras.saving.register_keras_serializable(package="PQuant")
class QuantizedReLU(keras.layers.Layer):
    def __init__(self, config, i, f, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.config = config
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)
        self.k = convert_to_tensor(0.0)

        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.use_multiplier = config.quantization_parameters.use_relu_multiplier
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress


        self.post_fitcompress_calibration = False
        self.saved_inputs = []
    

    def build(self, input_shape):
        super().build(input_shape)
        self.quantizer = create_quantizer(
            k=self.k,
            i=self.i,
            f=self.f,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        if self.use_hgq:
            self.quantizer.build(input_shape)

        if self.use_multiplier:
            self.multiplier = self.add_weight(shape=(1,), trainable=True, initializer=keras.initializers.Constant(-1.0))

    def set_bits(self, i, f):
        self.i = convert_to_tensor(i)
        self.f = convert_to_tensor(f)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def hgq_loss(self):
        if self.is_pretraining:
            return 0.0
        return (
            ops.sum(self.hgq.quantizer.i) + ops.sum(self.hgq.quantizer.f)
        ) * self.config.quantization_parameters.hgq_gamma

    def call(self, x):
        if self.use_fitcompress and self.is_pretraining:
            if self.post_fitcompress_calibration:
                # Save quantized input into ReLU
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized activations
            return ops.relu(x)
        # Multiplier after fitcompress if condition, such that we don't use any relu multiplier during FITcompress search
        if self.use_multiplier:
            x = x * 2 ** (ops.stop_gradient(ops.round(self.multiplier) - self.multiplier) + self.multiplier)
        if self.use_hgq:
            return self.quantizer(x)
        return self.quantizer(x, k=self.k, i=self.i, f=self.f)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "i": float(self.i),
                "f": float(self.f),
            }
        )
        return config


def hard_sigmoid(x):
    """Computes hard_sigmoid function that saturates between 0 and 1."""
    x = 0.5 * x + 0.5
    x = maximum(x, 0.0)
    x = minimum(x, 1.0)
    return x


def hard_tanh(x):
    """Computes hard_tanh function that saturates between -1 and 1."""
    return 2.0 * hard_sigmoid(x) - 1.0
