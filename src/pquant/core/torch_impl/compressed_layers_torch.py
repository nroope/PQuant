import typing
from typing import Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.nn.common_types import _size_1_t, _size_2_t

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.utils import get_pruning_layer

if typing.TYPE_CHECKING:
    from pquant.core.torch_impl.fit_compress import call_fitcompress  # noqa: 401

from keras import ops

from pquant.core.quantizer_functions import create_quantizer

T = TypeVar("T")


class PQWeightBiasBase(nn.Module):
    def __init__(
        self,
        config,
        layer_type,
        quantize_input=True,
        quantize_output=False,
        input_quantization_bits: Tuple[T, T, T] = None,
        weight_quantization_bits: Tuple[T, T, T] = None,
        bias_quantization_bits: Tuple[T, T, T] = None,
        output_quantization_bits: Tuple[T, T, T] = None,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if input_quantization_bits is not None:
            self.k_input, self.i_input, self.f_input = input_quantization_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if weight_quantization_bits is not None:
            self.k_weight, self.i_weight, self.f_weight = weight_quantization_bits
        else:
            self.k_weight = config.quantization_parameters.default_weight_keep_negatives
            self.i_weight = config.quantization_parameters.default_weight_integer_bits
            self.f_weight = config.quantization_parameters.default_weight_fractional_bits
        if bias_quantization_bits is not None:
            self.k_bias, self.i_bias, self.f_bias = bias_quantization_bits
        else:
            self.k_bias = config.quantization_parameters.default_weight_keep_negatives
            self.i_bias = config.quantization_parameters.default_weight_integer_bits
            self.f_bias = config.quantization_parameters.default_weight_fractional_bits

        if output_quantization_bits is not None:
            self.k_output, self.i_output, self.f_output = output_quantization_bits
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

    def build(self, input_shape):
        # Build function to delay quantizer creation until after custom i,f bits have been set
        self.input_quantizer = PyTorchQuantizer(
            torch.tensor(self.k_input),
            torch.tensor(self.i_input),
            torch.tensor(self.f_input),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            True,
            self.hgq_gamma,
        )
        self.weight_quantizer = PyTorchQuantizer(
            torch.tensor(self.k_weight),
            torch.tensor(self.i_weight),
            torch.tensor(self.f_weight),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            False,
            self.hgq_gamma,
        )

        self.bias_quantizer = PyTorchQuantizer(
            torch.tensor(self.k_bias),
            torch.tensor(self.i_bias),
            torch.tensor(self.f_bias),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            False,
            self.hgq_gamma,
        )

        self.output_quantizer = PyTorchQuantizer(
            torch.tensor(self.k_output),
            torch.tensor(self.i_output),
            torch.tensor(self.f_output),
            self.overflow,
            self.round_mode,
            self.use_hgq,
            True,
            self.hgq_gamma,
        )

        self.n_parallel = ops.prod(tuple(input_shape)[1:-1])
        self.parallelization_factor = self.parallelization_factor if self.parallelization_factor > 0 else self.n_parallel
        self.built = True
        self.input_shape = input_shape

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def apply_final_compression(self):
        pass

    def save_weights(self):
        self.init_weight = self.weight.clone()

    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def ebops(self):
        return 0.0

    def hgq_loss(self):
        if self.pruning_layer.is_pretraining or not self.use_hgq:
            return 0.0
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        if self.bias is not None:
            loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def quantize(self, x, quantizer):
        if self.enable_quantization:
            return quantizer(x) if x is not None else x
        return x

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.pruning_layer(weight)
        return weight

    def is_fitcompress_pretraining(self):
        return self.pruning_layer.is_pretraining and self.use_fitcompress

    def pre_forward(self, x):
        if not self.built:
            self.build(x.shape)
        if self.quantize_input:
            if self.use_hgq and not self.input_quantizer.quantizer.built:
                self.input_quantizer.quantizer.build(x.shape)
            if not self.is_fitcompress_pretraining():
                x = self.input_quantizer(x)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        return x

    def post_forward(self, x):
        if self.quantize_output:
            if self.use_hgq and not self.output_quantizer.quantizer.built:
                self.output_quantizer.quantizer.build(x.shape)
            if not self.is_fitcompress_pretraining():
                x = self.output_quantizer(x)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x


class PQDense(PQWeightBiasBase, nn.Linear):
    def __init__(
        self,
        config,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_input=True,
        quantize_output=False,
        device=None,
        dtype=None,
        input_quantization_bits: Tuple[T, T, T] = None,
        weight_quantization_bits: Tuple[T, T, T] = None,
        bias_quantization_bits: Tuple[T, T, T] = None,
        output_quantization_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            config=config,
            layer_type="linear",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            input_quantization_bits=input_quantization_bits,
            weight_quantization_bits=weight_quantization_bits,
            bias_quantization_bits=bias_quantization_bits,
            output_quantization_bits=output_quantization_bits,
            **kwargs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.is_pretraining = True
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
        else:
            self.register_parameter("_bias", None)
        del self._parameters["weight"]
        del self._parameters["bias"]
        self.pruning_layer.build(self._weight.shape)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def ebops(self):
        bw_inp = self.input_quantizer.quantizer.bits_(self.input_shape)
        bw_ker = self.weight_quantizer.quantizer.bits_(ops.shape(self.weight))
        ebops = ops.sum(F.linear(bw_inp, bw_ker))
        ebops = ebops * self.n_parallel / self.parallelization_factor
        if self.bias is not None:
            bw_bias = self.bias_quantizer.quantizer.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(list(self.input_shape)), self.weight.dtype)
            ebops += ops.mean(bw_bias) * size
        return ebops

    @property
    def weight(self):
        if self.final_compression_done or self.is_fitcompress_pretraining():
            return self._weight
        if self.pruning_first:
            weight = self.prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done or self.is_fitcompress_pretraining():
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done = True

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self.post_forward(x)
        return x

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


class PQConv2d(PQWeightBiasBase, nn.Conv2d):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            **kwargs,
        )
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.is_pretraining = True
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
        else:
            self.register_parameter("_bias", None)
        del self._parameters["weight"]
        del self._parameters["bias"]
        self.pruning_layer.build(self._weight.shape)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def ebops(self):
        bw_inp = self.input_quantizer.quantizer.bits_(self.input_shape)
        bw_ker = self.weight_quantizer.quantizer.bits_(ops.shape(self.weight))
        if self.parallelization_factor < 0:
            ebops = ops.sum(F.conv2d(bw_inp, bw_ker, stride=self.stride, padding=self.padding, dilation=self.dilation))
        else:
            reduce_axis_kernel = tuple(range(2, 4))
            reduce_axis_input = (0,) + tuple(range(2, 4))

            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)
            ebops = ops.sum(bw_inp[None, :] * bw_ker)
        if self.bias is not None:
            size = ops.cast(ops.prod(list(self.input_shape)), self.weight.dtype)
            bw_bias = self.bias_quantizer.quantizer.bits_(ops.shape(self.bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    @property
    def weight(self):
        if self.final_compression_done:
            return self._weight
        if self.pruning_first:
            weight = self.prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done:
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done = True

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self.post_forward(x)
        return x

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self._bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += ", self.quantize_input={quantize_input} "
        s += ", self.quantize_output={quantize_output}"

        return s.format(**self.__dict__)


class PQConv1d(PQWeightBiasBase, nn.Conv1d):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            **kwargs,
        )
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.is_pretraining = True
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
        else:
            self.register_parameter("_bias", None)
        del self._parameters["weight"]
        del self._parameters["bias"]
        self.pruning_layer.build(self._weight.shape)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def ebops(self):
        bw_inp = self.input_quantizer.quantizer.bits_(self.input_shape)
        bw_ker = self.weight_quantizer.quantizer.bits_(ops.shape(self.weight))
        if self.parallelization_factor < 0:
            ebops = ops.sum(F.conv1d(bw_inp, bw_ker, stride=self.stride, padding=self.padding, dilation=self.dilation))
        else:
            reduce_axis_kernel = tuple(range(2, 3))
            reduce_axis_input = (0,) + tuple(range(2, 3))

            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)
            ebops = ops.sum(bw_inp[None, :] * bw_ker)
        if self.bias is not None:
            size = ops.cast(ops.prod(list(self.input_shape)), self.weight.dtype)
            bw_bias = self.bias_quantizer.quantizer.bits_(ops.shape(self.bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    @property
    def weight(self):
        if self.final_compression_done:
            return self._weight
        if self.pruning_first:
            weight = self.prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done:
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done = True

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self.post_forward(x)
        return x

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self._bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += ", self.quantize_input={quantize_input}"
        s += ", self.quantize_output={quantize_output}"
        return s.format(**self.__dict__)


def add_compression_layers_torch(model, config, input_shape, device="cuda"):
    model = add_quantized_activations_to_model_layer(model, config)
    # model = add_quantized_activations_to_model_functional(model, config)
    model = add_pruning_to_model(model, config)
    model = disable_pruning_from_layers(model, config)
    model = add_layer_specific_quantization_to_model(model, config)
    model.to(device)
    model(torch.rand(input_shape, device=next(model.parameters()).device))
    return model


class PQAvgPoolBase(nn.Module):

    def __init__(
        self,
        config,
        quantize_input=True,
        quantize_output=False,
        input_quantization_bits: Tuple[T, T, T] = None,
        output_quantization_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if input_quantization_bits is not None:
            self.k_input, self.i_input, self.f_input = input_quantization_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if output_quantization_bits is not None:
            self.k_output, self.i_output, self.f_output = output_quantization_bits
        else:
            self.k_output = config.quantization_parameters.default_data_keep_negatives
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits
        self.overflow = config.quantization_parameters.overflow
        self.config = config
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

    def build(self, input_shape):
        self.input_quantizer = PyTorchQuantizer(
            k=torch.tensor(self.k_input),
            i=torch.tensor(self.i_input),
            f=torch.tensor(self.f_input),
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        self.output_quantizer = PyTorchQuantizer(
            k=torch.tensor(self.k_output),
            i=torch.tensor(self.i_output),
            f=torch.tensor(self.f_output),
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        self.input_shape = input_shape

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def post_pretrain_function(self):
        self.is_pretraining = False

    def ebops(self):
        bw_inp = self.input_quantizer.quantizer.bits_(self.input_shape)
        return torch.sum(bw_inp)

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    def pre_pooling(self, x):
        if not hasattr(self, "input_quantizer"):
            self.build(x.shape)
        if self.is_fitcompress_pretraining():
            if self.post_fitcompress_calibration:
                # Save inputs
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized pooling
            return super().forward()
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x)
        return x

    def post_pooling(self, x):
        if self.quantize_output and self.enable_quantization and not self.is_fitcompress_pretraining():
            x = self.output_quantizer(x)
        return x

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, quantize_input={self.quantize_input}, quantize_output={self.quantize_output}"  # noqa: 501


class PQAvgPool1d(PQAvgPoolBase, nn.AvgPool1d):

    def __init__(
        self,
        config,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        quantize_input=True,
        quantize_output=False,
        input_quantization_bits: Tuple[T, T, T] = None,
        output_quantization_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            input_quantization_bits=input_quantization_bits,
            output_quantization_bits=output_quantization_bits,
            **kwargs,
        )

    def forward(self, x):
        x = self.pre_pooling(x)
        x = super().forward(x)
        x = self.post_pooling(x)
        return x


class PQAvgPool2d(PQAvgPoolBase, nn.AvgPool2d):

    def __init__(
        self,
        config,
        kernel_size: _size_2_t,
        stride: _size_2_t = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        quantize_input=True,
        quantize_output=False,
        input_quantization_bits: Tuple[T, T, T] = None,
        output_quantization_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            input_quantization_bits=input_quantization_bits,
            output_quantization_bits=output_quantization_bits,
            **kwargs,
        )

    def forward(self, x):
        x = self.pre_pooling(x)
        x = super().forward(x)
        x = self.post_pooling(x)
        return x


class PQBatchNorm2d(nn.BatchNorm2d):

    def __init__(
        self,
        config,
        num_features: int,
        eps: float = 1e-5,
        momentum: typing.Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        quantize_input=True,
        input_quantization_bits: Tuple[T, T, T] = None,
        weight_quantization_bits: Tuple[T, T, T] = None,
        bias_quantization_bits: Tuple[T, T, T] = None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        if input_quantization_bits is not None:
            self.k_input, self.i_input, self.f_input = input_quantization_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if weight_quantization_bits is not None:
            self.k_weight, self.i_weight, self.f_weight = weight_quantization_bits
        else:
            self.k_weight = config.quantization_parameters.default_weight_keep_negatives
            self.i_weight = config.quantization_parameters.default_weight_integer_bits
            self.f_weight = config.quantization_parameters.default_weight_fractional_bits
        if bias_quantization_bits is not None:
            self.k_bias, self.i_bias, self.f_bias = bias_quantization_bits
        else:
            self.k_bias = config.quantization_parameters.default_weight_keep_negatives
            self.i_bias = config.quantization_parameters.default_weight_integer_bits
            self.f_bias = config.quantization_parameters.default_weight_fractional_bits
        self.overflow = config.quantization_parameters.overflow
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.config = config
        self.quantize_input = quantize_input
        self._weight = nn.Parameter(self.weight.clone())
        self._bias = nn.Parameter(self.bias.clone())
        del self._parameters["weight"]
        del self._parameters["bias"]
        self.built = False
        self.final_compression_done = False
        self.is_pretraining = True

    def build(self, input_shape):
        self.built = True
        self.input_quantizer = PyTorchQuantizer(
            k=torch.tensor(self.k_input),
            i=torch.tensor(self.i_input),
            f=torch.tensor(self.f_input),
            overflow=self.overflow,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
        )
        self.weight_quantizer = PyTorchQuantizer(
            k=torch.tensor(self.k_weight),
            i=torch.tensor(self.i_weight),
            f=torch.tensor(self.f_weight),
            round_mode=self.round_mode,
            overflow=self.overflow,
            is_data=False,
            is_heterogeneous=self.use_hgq,
        )
        self.bias_quantizer = PyTorchQuantizer(
            k=torch.tensor(self.k_bias),
            i=torch.tensor(self.i_bias),
            f=torch.tensor(self.f_bias),
            round_mode=self.round_mode,
            overflow=self.overflow,
            is_data=False,
            is_heterogeneous=self.use_hgq,
        )
        shape = [1] * len(input_shape)
        shape[1] = input_shape[1]
        self._shape = tuple(shape)
        self.input_shape = input_shape

    def apply_final_compression(self):
        self.final_compression_done = True

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    @property
    def weight(self):
        if self.enable_quantization and not self.final_compression_done and not self.is_fitcompress_pretraining():
            return self.weight_quantizer(self._weight)
        return self._weight

    @property
    def bias(self):
        if self.enable_quantization and not self.final_compression_done and not self.is_fitcompress_pretraining():
            return self.bias_quantizer(self._bias)
        return self._bias

    def ebops(self):
        bw_inp = self.input_quantizer.quantizer.bits_(self.input_shape)
        bw_ker = ops.reshape(self.weight_quantizer.quantizer.bits_(self.running_mean.shape), self._shape)
        bw_bias = ops.reshape(self.bias_quantizer.quantizer.bits_(self.running_mean.shape), self._shape)
        size = ops.cast(ops.prod(list(self.input_shape)), self._weight.dtype)
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

    def post_pretrain_function(self):
        self.is_pretraining = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.built:
            self.build(input.shape)
        if self.quantize_input and self.enable_quantization:
            if self.use_hgq and not self.input_quantizer.quantizer.built:
                self.input_quantizer.quantizer.build(input.shape)
            if not self.is_fitcompress_pretraining():
                input = self.input_quantizer(input)
        return super().forward(input)


class QuantizedActivation(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def hgq_loss(self):
        return self.activation.hgq_loss()

    def forward(self, x):
        return self.activation(x)


class PyTorchQuantizer(nn.Module):
    # HGQ quantizer wrapper
    def __init__(self, k, i, f, overflow, round_mode, is_heterogeneous, is_data, hgq_gamma=0):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(k), requires_grad=False)
        self.i = torch.nn.Parameter(torch.tensor(i), requires_grad=False)
        self.f = torch.nn.Parameter(torch.tensor(f), requires_grad=False)
        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.quantizer = create_quantizer(self.k, self.i, self.f, overflow, round_mode, is_heterogeneous, is_data)
        self.is_pretraining = False
        self.hgq_gamma = hgq_gamma

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.quantizer.k, self.quantizer.quantizer.i, self.quantizer.quantizer.f
        else:
            return self.k, self.i, self.f

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.quantizer._i.assign(self.quantizer.quantizer._i * 0.0 + i)
            self.quantizer.quantizer._f.assign(self.quantizer.quantizer._f * 0.0 + f)
        self.i.data = i
        self.f.data = f

    def post_pretrain(self):
        self.is_pretraining = True

    def forward(self, x):
        if self.use_hgq:
            x = self.quantizer(x)
        else:
            x = self.quantizer(x, k=self.k, i=self.i, f=self.f)
        return x

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = (torch.sum(self.quantizer.quantizer.i) + torch.sum(self.quantizer.quantizer.f)) * self.hgq_gamma
        return loss


def add_layer_specific_quantization_to_model(module, config):
    for name, layer in module.named_modules():
        if isinstance(layer, PQWeightBiasBase):
            if name in config.quantization_parameters.layer_specific:
                layer_config = config.quantization_parameters.layer_specific[name]
                if "weight" in layer_config:
                    weight_int_bits = layer_config["weight"]["integer_bits"]
                    weight_fractional_bits = layer_config["weight"]["fractional_bits"]
                    layer.i_weight = torch.tensor(weight_int_bits)
                    layer.f_weight = torch.tensor(weight_fractional_bits)
                if "bias" in layer_config:
                    bias_int_bits = layer_config["bias"]["integer_bits"]
                    bias_fractional_bits = layer_config["bias"]["fractional_bits"]
                    layer.i_bias = torch.tensor(bias_int_bits)
                    layer.f_bias = torch.tensor(bias_fractional_bits)
                if "input" in layer_config:
                    if "integer_bits" in layer_config["input"]:
                        input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                        layer.i_input = input_int_bits
                    if "fractional_bits" in layer_config["input"]:
                        input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                        layer.f_input = input_fractional_bits
                    if "quantize" in layer_config["input"]:
                        quantize = layer_config["input"]["quantize"]
                        layer.quantize_input = quantize
                if "output" in layer_config:
                    if "integer_bits" in layer_config["output"]:
                        output_int_bits = torch.tensor(layer_config["output"]["integer_bits"])
                        layer.i_output = input_int_bits
                    if "fractional_bits" in layer_config["output"]:
                        input_fractional_bits = torch.tensor(layer_config["output"]["fractional_bits"])
                        layer.f_output = input_fractional_bits
                    if "quantize" in layer_config["output"]:
                        quantize = layer_config["output"]["quantize"]
                        layer.quantize_output = quantize

        elif layer.__class__ in [PQBatchNorm2d]:
            if name in config.quantization_parameters.layer_specific:
                layer_config = config.quantization_parameters.layer_specific[name]
                if "weight" in layer_config:
                    i = torch.tensor(layer_config["weight"]["integer_bits"])
                    f = torch.tensor(layer_config["weight"]["fractional_bits"])
                    layer.i_weight = i
                    layer.f_weight = f
                if "bias" in layer_config:
                    i = torch.tensor(layer_config["bias"]["integer_bits"])
                    f = torch.tensor(layer_config["bias"]["fractional_bits"])
                    layer.i_bias = i
                    layer.f_biast = f
                if "input" in layer_config:
                    if "integer_bits" in layer_config["input"]:
                        input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                        layer.i_input = input_int_bits
                    if "fractional_bits" in layer_config["input"]:
                        input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                        layer.f_input = input_fractional_bits
                    if "quantize" in layer_config["input"]:
                        quantize = layer_config["input"]["quantize"]
                        layer.quantize_input = quantize
        elif layer.__class__ in [PQAvgPool1d, PQAvgPool2d]:
            if name in config.quantization_parameters.layer_specific:
                layer_config = config.quantization_parameters.layer_specific[name]
                if "input" in layer_config:
                    if "integer_bits" in layer_config["input"]:
                        input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                        layer.i_input = input_int_bits
                    if "fractional_bits" in layer_config["input"]:
                        input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                        layer.f_input = input_fractional_bits
                    if "quantize" in layer_config["input"]:
                        quantize = layer_config["input"]["quantize"]
                        layer.quantize_input = quantize
                if "output" in layer_config:
                    if "integer_bits" in layer_config["output"]:
                        output_int_bits = torch.tensor(layer_config["output"]["integer_bits"])
                        layer.i_output = output_int_bits
                    if "fractional_bits" in layer_config["output"]:
                        output_fractional_bits = torch.tensor(layer_config["output"]["fractional_bits"])
                        layer.f_output = output_fractional_bits
                    if "quantize" in layer_config["output"]:
                        quantize = layer_config["output"]["quantize"]
                        layer.quantize_output = quantize

        elif layer.__class__ == QuantizedActivation:
            if name in config.quantization_parameters.layer_specific:
                layer_config = config.quantization_parameters.layer_specific[name]
                if "input" in layer_config:
                    if "integer_bits" in layer_config["input"]:
                        input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                        layer.activation.i_input = input_int_bits
                    if "fractional_bits" in layer_config["input"]:
                        input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                        layer.activation.f_input = input_fractional_bits
                    if "quantize" in layer_config["input"]:
                        quantize = layer_config["input"]["quantize"]
                        layer.activation.quantize_input = quantize
                if "output" in layer_config:
                    if "integer_bits" in layer_config["output"]:
                        output_int_bits = torch.tensor(layer_config["output"]["integer_bits"])
                        layer.activation.i_output = output_int_bits
                    if "fractional_bits" in layer_config["output"]:
                        output_fractional_bits = torch.tensor(layer_config["output"]["fractional_bits"])
                        layer.activation.f_output = output_fractional_bits
                    if "quantize" in layer_config["output"]:
                        quantize = layer_config["output"]["quantize"]
                        layer.activation.quantize_output = quantize
    return module


def add_quantized_activations_to_model_layer(module, config):
    if not config.quantization_parameters.enable_quantization:
        return module
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    # Replaces ReLU and Tanh layers with quantized versions
    for name, layer in module.named_children():
        i = config.quantization_parameters.default_data_integer_bits
        f = config.quantization_parameters.default_data_fractional_bits
        if layer.__class__ in [nn.ReLU]:
            # For ReLU, if using default values, add 1 bit since values are unsigned.
            # Otherwise user provides bits. TODO: Find better way to do this
            f = config.quantization_parameters.default_data_fractional_bits + 1
            relu = QuantizedActivation(QuantizedReLU(config, i_input=i, f_input=f, i_output=i, f_output=f))
            setattr(module, name, relu)
        elif layer.__class__ in [nn.Tanh]:
            tanh = QuantizedActivation(QuantizedTanh(config, i_input=i, f_input=f, i_output=i, f_output=f))
            setattr(module, name, tanh)
        elif layer.__class__ == nn.AvgPool1d:
            new_layer = PQAvgPool1d(
                config,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.ceil_mode,
                layer.count_include_pad,
                quantize_input,
                quantize_output,
            )
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.AvgPool2d:
            new_layer = PQAvgPool2d(
                config,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.ceil_mode,
                layer.count_include_pad,
                layer.divisor_override,
                quantize_input,
                quantize_output,
            )
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.BatchNorm2d:
            new_layer = PQBatchNorm2d(
                config,
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                quantize_input=quantize_input,
            )
            setattr(module, name, new_layer)
        else:
            layer = add_quantized_activations_to_model_layer(layer, config)
    return module


def add_quantized_activations_to_model_functional(module, config):
    # Currently not in use. TODO: Fix this
    if config.quantization_parameters.use_high_granularity_quantization:
        return module
    # Replaces functional activation calls with quantized versions
    traced_model = symbolic_trace(module)
    for node in traced_model.graph.nodes:
        if node.op in ["call_method", "call_function"] and (node.target == "tanh" or "function relu" in str(node.target)):
            with traced_model.graph.inserting_after(node):
                if node.name in config.quantization_parameters.layer_specific:
                    bits = config.quantization_parameters.layer_specific[node.name]["bits"]
                else:
                    bits = (
                        config.quantization_parameters.default_integer_bits
                        + config.quantization_parameters.default_fractional_bits
                        + 1
                    )  # 1 sign bit
                kwargs = {"bits": bits}
                if node.target == "tanh":
                    kwargs["use_real_tanh"] = config.quantization_parameters.use_real_tanh
                    kwargs["use_symmetric"] = config.quantization_parameters.use_symmetric_quantization
                    # new_node = traced_model.graph.call_function(quantized_tanh, node.args, kwargs)
                else:
                    kwargs = {"integer_bits": config.quantization_parameters.default_integer_bits, "bits": bits}
                    # new_node = traced_model.graph.call_function(quantized_relu, node.args, kwargs)
                # node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)

    traced_model.graph.lint()
    traced_model.recompile()
    return traced_model


def disable_pruning_from_layers(module, config):
    for name, layer in module.named_modules():
        enable_pruning = name not in config.pruning_parameters.disable_pruning_for_layers
        if layer.__class__ in [PQDense, PQConv2d, PQConv1d] and not enable_pruning:
            layer.enable_pruning = enable_pruning
    return module


def add_pruning_to_model(module, config):
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    for name, layer in module.named_children():
        if layer.__class__ is nn.Linear:
            sparse_layer = PQDense(
                config, layer.in_features, layer.out_features, layer.bias is not None, quantize_input, quantize_output
            )
            sparse_layer.pruning_layer.build(layer.weight.shape)
            sparse_layer._weight.data = layer.weight.data
            if layer.bias is not None:
                sparse_layer._bias.data = layer.bias.data

            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv2d:
            sparse_layer = PQConv2d(
                config,
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer.bias is not None,
                layer.padding_mode,
                layer.weight.device,
                layer.weight.dtype,
                quantize_input,
                quantize_output,
            )
            sparse_layer.pruning_layer.build(layer.weight.shape)
            sparse_layer._weight.data = layer.weight.data
            if layer.bias is not None:
                sparse_layer._bias.data = layer.bias.data
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv1d:
            sparse_layer = PQConv1d(
                config,
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer.bias is not None,
                layer.padding_mode,
                layer.weight.device,
                layer.weight.dtype,
                quantize_input,
                quantize_output,
            )
            sparse_layer.pruning_layer.build(layer.weight.shape)
            sparse_layer._weight.data = layer.weight.data
            if layer.bias is not None:
                sparse_layer._bias.data = layer.bias.data
            setattr(module, name, sparse_layer)
        else:
            add_pruning_to_model(layer, config)
    return module


def apply_final_compression_torch(module):
    for layer in module.modules():
        if isinstance(layer, (PQWeightBiasBase, PQBatchNorm2d)):
            layer.apply_final_compression()
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
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.save_weights()


def rewind_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.rewind_weights()


def pre_finetune_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config, train_loader=None, loss_func=None):

    if config.fitcompress_parameters.enable_fitcompress:
        from pquant.core.torch_impl.fit_compress import call_fitcompress  # noqa: 811

        config, pruning_mask_importance_scores = call_fitcompress(config, model, train_loader, loss_func)

    # idx = 0
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.post_pre_train_function()
            layer.post_pre_train_function()

            # layer.pruning_layer.mask = pruning_mask_importance_scores[idx]
            # idx += 1

        elif isinstance(layer, QuantizedActivation):
            layer.activation.post_pre_train_function()
        elif isinstance(layer, (PQBatchNorm2d, PQAvgPool1d, PQAvgPool2d)):
            layer.post_pretrain_function()
    if config.pruning_parameters.pruning_method == "pdp" or (
        config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
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
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            if global_weights is None:
                global_weights = layer._weight.flatten()
            else:
                global_weights = torch.concat((global_weights, layer._weight.flatten()))

    abs_global_weights = torch.abs(global_weights)
    global_weight_topk, _ = torch.topk(abs_global_weights, abs_global_weights.numel())
    threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * global_weight_topk.numel())]
    global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            weight_size = layer._weight.numel()
            w = torch.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = w / weight_size
            layer.pruning_layer.sparsity = w / weight_size  # Wanda
            idx += weight_size


@torch.no_grad
def get_layer_keep_ratio_torch(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            weight, _ = layer.weight, layer.bias
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
    loss = 0.0
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            if layer.enable_pruning:
                loss += layer.pruning_layer.calculate_additional_loss()
            if layer.use_hgq:
                loss += layer.hgq_loss()
            losses += loss
        elif isinstance(layer, (QuantizedActivation)):
            if layer.activation.use_hgq:
                losses += layer.hgq_loss()
        elif isinstance(layer, (PQAvgPool1d, PQBatchNorm2d)):
            if layer.use_hgq:
                losses += layer.hgq_loss()
    return losses


def create_default_layer_quantization_pruning_config(model):
    config = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for name, layer in model.named_modules():
        if layer.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            if layer.bias is None:
                config.layer_specific[name] = {
                    "input": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                    "weight": {"integer_bits": 0, "fractional_bits": 7},
                    "output": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                }
            else:
                config.layer_specific[name] = {
                    "input": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                    "weight": {"integer_bits": 0, "fractional_bits": 7},
                    "bias": {"integer_bits": 0, "fractional_bits": 7},
                    "output": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                }
            config.disable_pruning_for_layers.append(name)
        elif layer.__class__ in [nn.Tanh, nn.ReLU, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            config.layer_specific[name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
            }
        elif layer.__class__ in [nn.BatchNorm2d]:
            config.layer_specific[name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "weight": {"integer_bits": 0, "fractional_bits": 7.0},
                "bias": {"integer_bits": 0, "fractional_bits": 7.0},
            }
    return config


def add_default_layer_quantization_pruning_to_config_torch(model, config):
    custom_scheme = create_default_layer_quantization_pruning_config(model)
    config.quantization_parameters.layer_specific = custom_scheme["layer_specific"]
    config.pruning_parameters.disable_pruning_for_layers = custom_scheme["disable_pruning_for_layers"]
    return config
