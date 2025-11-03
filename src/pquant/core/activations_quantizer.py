import keras
from keras import ops
from keras.ops import convert_to_tensor, maximum, minimum, tanh

from pquant.core.quantizer_functions import create_quantizer


@keras.saving.register_keras_serializable(package="PQuant")
class QuantizedTanh(keras.layers.Layer):
    def __init__(
        self, config, i_input=0.0, f_input=7.0, i_output=0.0, f_output=7.0, quantize_input=True, quantize_output=False
    ):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.i_input = convert_to_tensor(i_input)
        self.f_input = convert_to_tensor(f_input)
        self.k = convert_to_tensor(1.0)

        self.i_output = convert_to_tensor(i_output)
        self.f_output = convert_to_tensor(f_output)
        self.k = convert_to_tensor(1.0)

        self.config = config

        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.use_real_tanh = config.quantization_parameters.use_real_tanh
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

    def build(self, input_shape):
        super().build(input_shape)
        self.input_shape = input_shape
        self.output_quantizer = create_quantizer(
            k=self.k,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        self.input_quantizer = create_quantizer(
            k=self.k,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.output_quantizer.build(input_shape)

    def get_input_quantization_bits(self):
        if self.use_hgq:
            return self.input_quantizer.quantizer.k, self.input_quantizer.quantizer.i, self.input_quantizer.quantizer.f
        else:
            return self.k, self.i_input, self.f_input

    def set_input_quantization_bits(self, i, f):
        if self.use_hgq:
            self.input_quantizer.quantizer._i.assign(self.input_quantizer.quantizer._i * 0.0 + i)
            self.input_quantizer.quantizer._f.assign(self.input_quantizer.quantizer._f * 0.0 + f)
        else:
            self.i_input = i
            self.f_input = f

    def get_output_quantization_bits(self):
        if self.use_hgq:
            return self.output_quantizer.quantizer.k, self.output_quantizer.quantizer.i, self.output_quantizer.quantizer.f
        else:
            return self.k, self.i_output, self.f_output

    def set_output_quantization_bits(self, i, f):
        if self.use_hgq:
            self.output_quantizer.quantizer._i.assign(self.output_quantizer.quantizer._i * 0.0 + i)
            self.output_quantizer.quantizer._f.assign(self.output_quantizer.quantizer._f * 0.0 + f)
        else:
            self.i_output = i
            self.f_output = f

    def ebops(self):
        bw_inp = self.input_quantizer.bits_(self.input_shape)
        bw_out = self.output_quantizer.bits_(self.input_shape)
        return ops.sum((2.0**bw_inp) * bw_out) * 1e-4  # type: ignore

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.beta * self.ebops()
        loss += (ops.sum(self.input_quantizer.quantizer.i) + ops.sum(self.input_quantizer.quantizer.f)) * self.hgq_gamma
        loss += (ops.sum(self.output_quantizer.quantizer.i) + ops.sum(self.output_quantizer.quantizer.f)) * self.hgq_gamma
        return loss

    def post_pre_train_function(self):
        self.is_pretraining = False

    def pre_activation(self, x):
        if self.quantize_input:
            if self.use_hgq:
                x = self.input_quantizer(x)
            else:
                x = self.input_quantizer(x, k=self.k, i=self.i_input, f=self.f_input)
        return x

    def post_activation(self, x):
        if self.quantize_output:
            if self.use_hgq:
                return self.output_quantizer(x)
            else:
                return self.output_quantizer(x, k=self.k, i=self.i_input, f=self.f_output)
        return x

    def call(self, x):
        x = self.pre_activation(x)
        x = tanh(x) if self.use_real_tanh else hard_tanh(x)
        x = self.post_activation(x)
        self.add_loss(self.hgq_loss())
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.get_dict(), "i": float(self.i), "f": float(self.f)})
        return config


@keras.saving.register_keras_serializable(package="PQuant")
class QuantizedReLU(keras.layers.Layer):
    def __init__(
        self, config, i_input=0.0, f_input=8.0, i_output=0.0, f_output=8.0, quantize_input=True, quantize_output=False
    ):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.config = config
        self.i_input = convert_to_tensor(i_input)
        self.f_input = convert_to_tensor(f_input)
        self.k = convert_to_tensor(0.0)

        self.i_output = convert_to_tensor(i_output)
        self.f_output = convert_to_tensor(f_output)
        self.k = convert_to_tensor(0.0)

        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow = config.quantization_parameters.overflow
        self.use_multiplier = config.quantization_parameters.use_relu_multiplier
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress

        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

    def build(self, input_shape):
        super().build(input_shape)
        self.input_shape = input_shape
        self.output_quantizer = create_quantizer(
            k=self.k,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        self.input_quantizer = create_quantizer(
            k=self.k,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_data=True,
            is_heterogeneous=self.use_hgq,
        )
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.output_quantizer.build(input_shape)

        if self.use_multiplier:
            self.multiplier = self.add_weight(shape=(1,), trainable=True, initializer=keras.initializers.Constant(-1.0))

    def get_input_quantization_bits(self):
        if self.use_hgq:
            return self.input_quantizer.quantizer.k, self.input_quantizer.quantizer.i, self.input_quantizer.quantizer.f
        else:
            return self.k, self.i_input, self.f_input

    def set_input_quantization_bits(self, i, f):
        if self.use_hgq:
            self.input_quantizer.quantizer._i.assign(self.input_quantizer.quantizer._i * 0.0 + i)
            self.input_quantizer.quantizer._f.assign(self.input_quantizer.quantizer._f * 0.0 + f)
        else:
            self.i_input = i
            self.f_input = f

    def get_output_quantization_bits(self):
        if self.use_hgq:
            return self.output_quantizer.quantizer.k, self.output_quantizer.quantizer.i, self.output_quantizer.quantizer.f
        else:
            return self.k, self.i_output, self.f_output

    def set_output_quantization_bits(self, i, f):
        if self.use_hgq:
            self.output_quantizer.quantizer._i.assign(self.output_quantizer.quantizer._i * 0.0 + i)
            self.output_quantizer.quantizer._f.assign(self.output_quantizer.quantizer._f * 0.0 + f)
        else:
            self.i_output = i
            self.f_output = f

    def post_pre_train_function(self):
        self.is_pretraining = False

    def ebops(self):
        bw_inp = self.input_quantizer.bits_(self.input_shape)
        bw_out = self.output_quantizer.bits_(self.input_shape)
        return ops.sum((2.0**bw_inp) * bw_out) * 1e-4  # type: ignore

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.beta * self.ebops()
        loss = self.beta * self.ebops()
        loss += (ops.sum(self.input_quantizer.quantizer.i) + ops.sum(self.input_quantizer.quantizer.f)) * self.hgq_gamma
        loss += (ops.sum(self.output_quantizer.quantizer.i) + ops.sum(self.output_quantizer.quantizer.f)) * self.hgq_gamma
        return loss

    def pre_activation(self, x):
        if self.quantize_input:
            if self.use_hgq:
                x = self.input_quantizer(x)
            else:
                x = self.input_quantizer(x, k=self.k, i=self.i_input, f=self.f_input)
        if self.use_multiplier:
            x = x * 2 ** (ops.stop_gradient(ops.round(self.multiplier) - self.multiplier) + self.multiplier)
        return x

    def post_activation(self, x):
        if self.quantize_output:
            if self.use_hgq:
                return self.output_quantizer(x)
            else:
                return self.output_quantizer(x, k=self.k, i=self.i_input, f=self.f_output)
        return x

    def call(self, x):
        if self.use_fitcompress and self.is_pretraining:
            if self.post_fitcompress_calibration:
                # Save quantized input into ReLU
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized activations
            return ops.relu(x)
        # Multiplier after fitcompress if condition, such that we don't use any relu multiplier during FITcompress search
        x = self.pre_activation(x)
        x = ops.relu(x)
        x = self.post_activation(x)
        self.add_loss(self.hgq_loss())
        return x

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
