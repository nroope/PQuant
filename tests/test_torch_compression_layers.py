import keras
import numpy as np
import pytest
import torch
from keras import ops
from torch import nn
from torch.nn import AvgPool2d, Conv1d, Conv2d, Linear, ReLU, Tanh

from pquant import post_training_prune
from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.torch_impl.compressed_layers_torch import (
    CompressedLayerBase,
    CompressedLayerConv1d,
    CompressedLayerConv2d,
    CompressedLayerLinear,
    QuantizedPooling,
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
            "hgq_heterogeneous": True,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
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
            "hgq_heterogeneous": True,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }


@pytest.fixture
def config_wanda():
    return {
        "pruning_parameters": {
            "calculate_pruning_budget": True,
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "pruning_method": "wanda",
            "sparsity": 0.75,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 2,
            "N": None,
            "M": None,
        },
        "quantization_parameters": {
            "default_integer_bits": 0.0,
            "default_fractional_bits": 7.0,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "hgq_heterogeneous": True,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
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
            "hgq_heterogeneous": True,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }


@pytest.fixture
def conv2d_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, 32, 32).astype(np.float32))


@pytest.fixture
def conv1d_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, 32).astype(np.float32))


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
    for m in module.children():
        if hasattr(m, "built"):
            is_built.append(m.built)
        is_built = check_keras_layer_is_built(m, is_built)
    return is_built


class TestModelWithAvgPool(nn.Module):
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
        self.avg = AvgPool2d(2)

    def forward(self, x):
        x = self.submodule(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.avg(x)
        return x


def test_hgq_activation_built(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModelWithAvgPool(layer, "relu")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    is_built = check_keras_layer_is_built(model, [])
    assert all(is_built)

    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModelWithAvgPool(layer, "tanh")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    is_built = check_keras_layer_is_built(model, [])
    assert all(is_built)


def test_post_training_wanda(config_wanda, conv2d_input):
    config_wanda["pruning_parameters"]["calculate_pruning_budget"] = False
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer, "relu")
    calibration_dataset = [conv2d_input, conv2d_input]
    model = post_training_prune(model, calibration_dataset, config_wanda)
    assert get_layer_keep_ratio_torch(model) == 1 - config_wanda["pruning_parameters"]["sparsity"]


class TestModel2(nn.Module):
    __test__ = False

    def __init__(self, submodule, submodule2, activation=None, activation2=None):
        super().__init__()
        self.submodule = submodule
        self.submodule2 = submodule2
        if activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            self.activation = activation

        if activation2 == "relu":
            self.activation2 = ReLU()
        elif activation2 == "tanh":
            self.activation2 = Tanh()
        else:
            self.activation2 = activation2

    def forward(self, x):
        x = self.submodule(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.submodule2(x)
        if self.activation2 is not None:
            x = self.activation2(x)
        return x


def test_calculate_pruning_budget(config_wanda, dense_input):
    sparsity = 0.75
    config_wanda["pruning_parameters"]["calculate_pruning_budget"] = True
    config_wanda["pruning_parameters"]["sparsity"] = sparsity

    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu")

    # First layer will have 50% sparsity
    weight = np.ones(IN_FEATURES * OUT_FEATURES).astype(np.float32)
    weight[: IN_FEATURES * OUT_FEATURES // 2] = 0.001
    weight = ops.convert_to_tensor(weight)
    weight2 = ops.linspace(0.01, 0.99, OUT_FEATURES * OUT_FEATURES)

    model = add_compression_layers_torch(model, config_wanda, dense_input.shape)
    model.submodule.weight.data = ops.reshape(weight, model.submodule.weight.shape)
    model.submodule2.weight.data = ops.reshape(weight2, model.submodule2.weight.shape)

    # Triggers calculation of pruning budget for PDP and Wanda
    post_pretrain_functions(model, config_wanda)
    total_weights = IN_FEATURES * OUT_FEATURES + OUT_FEATURES * OUT_FEATURES
    remaining_weights = 0
    for layer in model.modules():
        if hasattr(layer, "pruning_layer"):
            calculated_sparsity = layer.pruning_layer.sparsity.cpu()
            remaining_weights += np.float32(1 - calculated_sparsity) * layer.weight.numel()
    # First layer should have 50% sparsity, total sparsity should be around 75%
    assert model.submodule.pruning_layer.sparsity == 0.5
    np.testing.assert_allclose(remaining_weights / total_weights, 1 - sparsity, atol=1e-3, rtol=0)


def test_trigger_post_pretraining(config_pdp, dense_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu", "tanh")

    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)

    assert model.submodule.pruning_layer.is_pretraining is True
    assert model.activation.is_pretraining is True
    assert model.submodule2.pruning_layer.is_pretraining is True
    assert model.activation2.is_pretraining is True

    post_pretrain_functions(model, config_pdp)

    assert model.submodule.pruning_layer.is_pretraining is False
    assert model.activation.is_pretraining is False
    assert model.submodule2.pruning_layer.is_pretraining is False
    assert model.activation2.is_pretraining is False


def test_hgq_weight_shape(config_pdp, dense_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu", "tanh")

    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)

    assert model.submodule.hgq_weight.quantizer._i.shape == model.submodule.weight.shape
    assert model.activation.hgq.quantizer._i.shape == (1, OUT_FEATURES)

    config_pdp["quantization_parameters"]["hgq_heterogeneous"] = False
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu", "tanh")

    model = add_compression_layers_torch(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)

    assert model.submodule.hgq_weight.quantizer._i.shape == (1, 1)
    assert model.activation.hgq.quantizer._i.shape == (1, 1)


def test_set_activation_custom_bits_hgq(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    layer2 = AvgPool2d(2)
    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (CompressedLayerBase)):
            assert m.i_weight == 0.0
            assert m.i_bias == 0.0
            assert torch.all(m.hgq_weight.quantizer.i == 0.0)
            assert torch.all(m.hgq_bias.quantizer.i == 0.0)

            assert m.f_weight == 7.0
            assert m.f_bias == 7.0
            assert torch.all(m.hgq_weight.quantizer.f == 7.0)
            assert torch.all(m.hgq_bias.quantizer.f == 7.0)
        elif isinstance(m, (QuantizedTanh)):
            assert m.i == 0.0
            assert m.f == 7.0
            assert torch.all(m.hgq.quantizer.i == 0.0)
            assert torch.all(m.hgq.quantizer.f == 7.0)
        elif isinstance(m, (QuantizedReLU)):
            assert m.i == 0.0
            assert m.f == 8.0
            assert torch.all(m.hgq.quantizer.i == 0.0)
            assert torch.all(m.hgq.quantizer.f == 8.0)

        elif isinstance(m, QuantizedPooling):
            assert m.i == 0.0
            assert m.f == 7.0
            assert torch.all(m.hgq.quantizer.i == 0.0)
            assert torch.all(m.hgq.quantizer.f == 7.0)

    config_pdp["quantization_parameters"]["layer_specific"] = {
        'submodule': {
            'weight': {'integer_bits': 1, 'fractional_bits': 3},
            'bias': {'integer_bits': 2, 'fractional_bits': 4},
        },
        'submodule2': {'integer_bits': 1, 'fractional_bits': 3},
        'activation': {'integer_bits': 0, 'fractional_bits': 4},
        'activation2': {'integer_bits': 0, 'fractional_bits': 3},
    }

    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (CompressedLayerBase)):
            assert m.i_weight == 1.0
            assert m.i_bias == 2.0
            assert torch.all(m.hgq_weight.quantizer.i == 1.0)
            assert torch.all(m.hgq_bias.quantizer.i == 2.0)

            assert m.f_weight == 3.0
            assert m.f_bias == 4.0
            assert torch.all(m.hgq_weight.quantizer.f == 3.0)
            assert torch.all(m.hgq_bias.quantizer.f == 4.0)
        elif isinstance(m, (QuantizedTanh)):
            assert m.i == 0.0
            assert m.f == 3.0
            assert torch.all(m.hgq.quantizer.i == 0.0)
            assert torch.all(m.hgq.quantizer.f == 3.0)
        elif isinstance(m, (QuantizedReLU)):
            assert m.i == 0.0
            assert m.f == 4.0
            assert torch.all(m.hgq.quantizer.i == 0.0)
            assert torch.all(m.hgq.quantizer.f == 4.0)
        elif isinstance(m, QuantizedPooling):
            assert m.i == 1.0
            assert m.f == 3.0
            assert torch.all(m.hgq.quantizer.i == 1.0)
            assert torch.all(m.hgq.quantizer.f == 3.0)


def test_set_activation_custom_bits_quantizer(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = False
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    layer2 = AvgPool2d(2)
    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (CompressedLayerBase)):
            assert m.i_weight == 0.0
            assert m.f_bias == 7.0
        elif isinstance(m, (QuantizedTanh)):
            assert m.i == 0.0
            assert m.f == 7.0
        elif isinstance(m, (QuantizedReLU)):
            assert m.i == 0.0
            assert m.f == 8.0

    config_pdp["quantization_parameters"]["layer_specific"] = {
        'submodule': {
            'weight': {'integer_bits': 1.0, 'fractional_bits': 3.0},
            'bias': {'integer_bits': 1.0, 'fractional_bits': 3.0},
        },
        'submodule2': {'integer_bits': 1.0, 'fractional_bits': 3.0},
        'activation': {'integer_bits': 0.0, 'fractional_bits': 4.0},
        'activation2': {'integer_bits': 0.0, 'fractional_bits': 3.0},
    }

    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers_torch(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (CompressedLayerBase)):
            assert m.i_weight == 1.0
            assert m.f_bias == 3.0
        elif isinstance(m, (QuantizedTanh)):
            assert m.i == 0.0
            assert m.f == 3.0
        elif isinstance(m, (QuantizedReLU)):
            assert m.i == 0.0
            assert m.f == 4.0
        elif isinstance(m, QuantizedPooling):
            assert m.i == 1.0
            assert m.f == 3.0
