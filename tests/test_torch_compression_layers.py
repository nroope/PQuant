import keras
import numpy as np
import pytest
import torch
from keras import ops
from torch import nn
from torch.nn import Conv1d, Conv2d, Linear, ReLU, Tanh

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.torch_impl.compressed_layers_torch import (
    CompressedLayerConv1d,
    CompressedLayerConv2d,
    CompressedLayerLinear,
    add_compression_layers_torch,
    get_layer_keep_ratio_torch,
    post_pretrain_functions,
    pre_finetune_functions,
    remove_pruning_from_model_torch,
)

BATCH_SIZE = 4
OUT_FEATURES = 32
IN_FEATURES = 16
KERNEL_SIZE = 3
STEPS = 16


@pytest.fixture
def config_pdp():
    return {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "epsilon": 1.0,
            "pruning_method": "pdp",
            "sparsity": 0.75,
            "temperature": 1e-5,
            "threshold_decay": 0.0,
            "structured_pruning": False,
        },
        "quantization_parameters": {
            "default_integer_bits": 0.0,
            "default_fractional_bits": 7.0,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
    }


@pytest.fixture
def config_ap():
    return {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "pruning_method": "activation_pruning",
            "threshold": 0.3,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 1,
        },
        "quantization_parameters": {
            "default_integer_bits": 0.0,
            "default_fractional_bits": 7.0,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
    }


@pytest.fixture
def config_wanda():
    return {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "pruning_method": "wanda",
            "sparsity": 0.75,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 1,
            "N": None,
            "M": None,
        },
        "quantization_parameters": {
            "default_integer_bits": 0.0,
            "default_fractional_bits": 7.0,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
    }


@pytest.fixture
def config_cs():
    return {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "final_temp": 200,
            "pruning_method": "cs",
            "threshold_decay": 0.0,
            "threshold_init": 0.1,
        },
        "quantization_parameters": {
            "default_integer_bits": 0.0,
            "default_fractional_bits": 7.0,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
    }


@pytest.fixture
def conv2d_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, KERNEL_SIZE, KERNEL_SIZE).astype(np.float32))


@pytest.fixture
def conv1d_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, STEPS).astype(np.float32))


@pytest.fixture
def dense_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES).astype(np.float32))


class TestModel(nn.Module):
    __test__ = False

    def __init__(self, submodule, activation=None):
        super().__init__()
        self.submodule = submodule
        if activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            self.activation = activation

    def forward(self, x):
        x = self.submodule(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def test_dense_call(config_pdp, dense_input):
    layer_to_replace = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    out = layer_to_replace(dense_input)
    layer = CompressedLayerLinear(config_pdp, layer_to_replace, "linear")
    layer.weight.data = layer_to_replace.weight.data
    out2 = layer(dense_input)
    assert ops.all(ops.equal(out, out2))


def test_conv2d_call(config_pdp, conv2d_input):
    layer_to_replace = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False, padding="same")
    out = layer_to_replace(conv2d_input)
    layer = CompressedLayerConv2d(config_pdp, layer_to_replace, "conv")
    layer.weight.data = layer_to_replace.weight.data
    out2 = layer(conv2d_input)
    assert ops.all(ops.equal(out, out2))


def test_conv1d_call(config_pdp, conv1d_input):
    layer_to_replace = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, stride=2, bias=False)
    out = layer_to_replace(conv1d_input)
    layer = CompressedLayerConv1d(config_pdp, layer_to_replace, "conv")
    layer.weight.data = layer_to_replace.weight.data
    out2 = layer(conv1d_input)
    assert ops.all(ops.equal(out, out2))


def test_dense_add_remove_layers(config_pdp, dense_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer)
    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    output1 = model(dense_input)
    model = remove_pruning_from_model_torch(model, config_pdp)
    output2 = model(dense_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.submodule.weight)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    output1 = model(conv2d_input)
    model = remove_pruning_from_model_torch(model, config_pdp)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.submodule.weight)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv1d_add_remove_layers(config_pdp, conv1d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers_torch(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.submodule.weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    output1 = model(conv1d_input)
    model = remove_pruning_from_model_torch(model, config_pdp)
    output2 = model(conv1d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.submodule.weight)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_dense_get_layer_keep_ratio(config_pdp, dense_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer)
    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    model(dense_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_torch(model)
    model = remove_pruning_from_model_torch(model, config_pdp)
    ratio2 = get_layer_keep_ratio_torch(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_torch(model)
    model = remove_pruning_from_model_torch(model, config_pdp)
    ratio2 = get_layer_keep_ratio_torch(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv1d_get_layer_keep_ratio(config_pdp, conv1d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers_torch(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.submodule.weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_torch(model)
    model = remove_pruning_from_model_torch(model, config_pdp)
    ratio2 = get_layer_keep_ratio_torch(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_check_activation(config_pdp, dense_input):
    # ReLU
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, ReLU)

    config_pdp["quantization_parameters"]["enable_quantization"] = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, QuantizedReLU)

    # Tanh
    config_pdp["quantization_parameters"]["enable_quantization"] = False
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "tanh")
    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, Tanh)

    config_pdp["quantization_parameters"]["enable_quantization"] = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "tanh")
    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, QuantizedTanh)


def check_keras_layer_is_built(module, is_built):
    for n, m in module.named_children():
        if isinstance(m, keras.layers.Layer):
            is_built.append(n)
        is_built = check_keras_layer_is_built(m, is_built)
    return is_built


def test_hgq_activation_built(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer, "relu")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    is_built = check_keras_layer_is_built(model, [])
    assert all(is_built)

    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer, "tanh")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    is_built = check_keras_layer_is_built(model, [])
    assert all(is_built)
