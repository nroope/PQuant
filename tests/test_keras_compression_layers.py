import keras
import numpy as np
import pytest
from keras import ops
from keras.api.layers import (
    Activation,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    ReLU,
    SeparableConv2D,
)

from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
from pquant.core.tf_impl.compressed_layers_tf import (
    CompressedLayerConv1dKeras,
    CompressedLayerConv2dKeras,
    CompressedLayerDenseKeras,
    CompressedLayerSeparableConv2dKeras,
    add_compression_layers_tf,
    get_layer_keep_ratio_tf,
    post_pretrain_functions,
    pre_finetune_functions,
    remove_pruning_from_model_tf,
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
            "hgq_heterogeneous": True,
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
            "calculate_pruning_budget": False,
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
            "hgq_heterogeneous": True,
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
            "hgq_heterogeneous": True,
            "layer_specific": [],
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_symmetric_quantization": False,
        },
        "training_parameters": {"pruning_first": False},
    }


@pytest.fixture
def conv2d_input():
    if keras.backend.image_data_format() == "channels_first":
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, KERNEL_SIZE, KERNEL_SIZE))
    else:
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, KERNEL_SIZE, KERNEL_SIZE, IN_FEATURES))
    return inp


@pytest.fixture
def conv1d_input():
    if keras.backend.image_data_format() == "channels_first":
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, STEPS))
    else:
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, STEPS, IN_FEATURES))
    return inp


@pytest.fixture
def dense_input():
    return ops.convert_to_tensor(np.random.rand(BATCH_SIZE, IN_FEATURES))


def test_dense_call(config_pdp, dense_input):
    layer_to_replace = Dense(OUT_FEATURES, use_bias=False)
    layer_to_replace.build((BATCH_SIZE, IN_FEATURES))
    out = layer_to_replace(dense_input)
    layer = CompressedLayerDenseKeras(config_pdp, layer_to_replace, "linear")
    layer.build(dense_input.shape)
    layer.weight.assign(layer_to_replace.kernel)
    out2 = layer(dense_input)
    assert ops.all(ops.equal(out, out2))


def test_conv2d_call(config_pdp, conv2d_input):
    layer_to_replace = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")
    layer_to_replace.build(conv2d_input.shape)
    out = layer_to_replace(conv2d_input)
    layer = CompressedLayerConv2dKeras(config_pdp, layer_to_replace, "conv")
    layer.build(conv2d_input.shape)
    layer.weight.assign(layer_to_replace.kernel)
    out2 = layer(conv2d_input)
    assert ops.all(ops.equal(out, out2))


def test_separable_conv2d_call(config_pdp, conv2d_input):
    layer_to_replace = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")
    layer_to_replace.build(conv2d_input.shape)
    out = layer_to_replace(conv2d_input)
    layer = CompressedLayerSeparableConv2dKeras(config_pdp, layer_to_replace)
    layer.depthwise_conv.build(conv2d_input.shape)
    layer.pointwise_conv.build(conv2d_input.shape)
    layer.depthwise_conv.weight.assign(layer_to_replace.depthwise_kernel)
    layer.pointwise_conv.weight.assign(layer_to_replace.pointwise_kernel)

    out2 = layer(conv2d_input)
    assert ops.all(ops.equal(out, out2))


def test_separable_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)

    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    # Set Depthwise mask to 50% 0s
    mask_50pct_dw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].depthwise_conv.weight)) < 0.5, "float32")
    mask_50pct_dw = ops.reshape(keras.random.shuffle(mask_50pct_dw), model.layers[1].depthwise_conv.pruning_layer.mask.shape)
    model.layers[1].depthwise_conv.pruning_layer.mask = mask_50pct_dw
    # Set Pointwise mask to 50% 0s
    mask_50pct_pw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].pointwise_conv.weight)) < 0.5, "float32")
    mask_50pct_pw = ops.reshape(keras.random.shuffle(mask_50pct_pw), model.layers[1].pointwise_conv.pruning_layer.mask.shape)
    model.layers[1].pointwise_conv.pruning_layer.mask = mask_50pct_pw

    output1 = model(conv2d_input)
    model = remove_pruning_from_model_tf(model, config_pdp)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))

    expected_nonzero_count_depthwise = ops.count_nonzero(mask_50pct_dw)
    nonzero_count_depthwise = ops.count_nonzero(model.layers[1].depthwise_kernel)
    assert ops.equal(expected_nonzero_count_depthwise, nonzero_count_depthwise)

    expected_nonzero_count_pointwise = ops.count_nonzero(mask_50pct_pw)
    nonzero_count_pointwise = ops.count_nonzero(model.layers[1].pointwise_kernel)
    assert ops.equal(expected_nonzero_count_pointwise, nonzero_count_pointwise)


def test_separable_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    # Set Depthwise mask to 50% 0s
    mask_50pct_dw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].depthwise_conv.weight)) < 0.5, "float32")
    mask_50pct_dw = ops.reshape(keras.random.shuffle(mask_50pct_dw), model.layers[1].depthwise_conv.pruning_layer.mask.shape)
    model.layers[1].depthwise_conv.pruning_layer.mask = mask_50pct_dw
    # Set Pointwise mask to 50% 0s
    mask_50pct_pw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].pointwise_conv.weight)) < 0.5, "float32")
    mask_50pct_pw = ops.reshape(keras.random.shuffle(mask_50pct_pw), model.layers[1].pointwise_conv.pruning_layer.mask.shape)
    model.layers[1].pointwise_conv.pruning_layer.mask = mask_50pct_pw

    ratio1 = get_layer_keep_ratio_tf(model)
    model = remove_pruning_from_model_tf(model, config_pdp)
    ratio2 = get_layer_keep_ratio_tf(model)

    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct_dw) / ops.size(mask_50pct_dw), ratio1)


def test_separable_conv2d_trigger_post_pretraining(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    act1 = Activation("tanh")(out)
    flat = keras.layers.Flatten()(act1)
    out2 = Dense(OUT_FEATURES, use_bias=False)(flat)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    assert model.layers[1].depthwise_conv.pruning_layer.is_pretraining is True
    assert model.layers[1].pointwise_conv.pruning_layer.is_pretraining is True
    assert model.layers[2].is_pretraining is True
    assert model.layers[4].pruning_layer.is_pretraining is True
    assert model.layers[5].is_pretraining is True

    post_pretrain_functions(model, config_pdp)

    assert model.layers[1].depthwise_conv.pruning_layer.is_pretraining is False
    assert model.layers[1].pointwise_conv.pruning_layer.is_pretraining is False
    assert model.layers[2].is_pretraining is False
    assert model.layers[4].pruning_layer.is_pretraining is False
    assert model.layers[5].is_pretraining is False


def test_conv1d_call(config_pdp, conv1d_input):
    layer_to_replace = Conv1D(OUT_FEATURES, KERNEL_SIZE, strides=2, use_bias=False)
    layer_to_replace.build(conv1d_input.shape)
    out = layer_to_replace(conv1d_input)
    layer = CompressedLayerConv1dKeras(config_pdp, layer_to_replace, "conv")
    layer.build(conv1d_input.shape)
    layer.weight.assign(layer_to_replace.kernel)
    out2 = layer(conv1d_input)
    assert ops.all(ops.equal(out, out2))


def test_dense_add_remove_layers(config_pdp, dense_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=(dense_input.shape[1:]))
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)
    model(dense_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(dense_input)
    model = remove_pruning_from_model_tf(model, config_pdp)
    output2 = model(dense_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(conv2d_input)
    model = remove_pruning_from_model_tf(model, config_pdp)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_depthwise_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(conv2d_input)
    model = remove_pruning_from_model_tf(model, config_pdp)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv1d_add_remove_layers(config_pdp, conv1d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model = add_compression_layers_tf(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(conv1d_input)
    model = remove_pruning_from_model_tf(model, config_pdp)
    output2 = model(conv1d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_dense_get_layer_keep_ratio(config_pdp, dense_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=(dense_input.shape[1:]))
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)
    model(dense_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_tf(model)
    model = remove_pruning_from_model_tf(model, config_pdp)
    ratio2 = get_layer_keep_ratio_tf(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_tf(model)
    model = remove_pruning_from_model_tf(model, config_pdp)
    ratio2 = get_layer_keep_ratio_tf(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_depthwise_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_tf(model)
    model = remove_pruning_from_model_tf(model, config_pdp)
    ratio2 = get_layer_keep_ratio_tf(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv1d_get_layer_keep_ratio(config_pdp, conv1d_input):
    config_pdp["pruning_parameters"]["enable_pruning"] = True
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model = add_compression_layers_tf(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio_tf(model)
    model = remove_pruning_from_model_tf(model, config_pdp)
    ratio2 = get_layer_keep_ratio_tf(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_check_activation(config_pdp, dense_input):
    # ReLU
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="relu")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)

    assert isinstance(model.layers[2], ReLU)

    config_pdp["quantization_parameters"]["enable_quantization"] = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="relu")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)
    assert isinstance(model.layers[2], QuantizedReLU)

    # Tanh
    config_pdp["quantization_parameters"]["enable_quantization"] = False
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="tanh")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)

    assert isinstance(model.layers[2], Activation)
    assert model.layers[2].activation.__name__ == "tanh"

    config_pdp["quantization_parameters"]["enable_quantization"] = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="tanh")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)
    assert isinstance(model.layers[2], QuantizedTanh)


def test_hgq_activation_built(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=True)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_conv2d_hgq")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)

    is_built = []
    for layer in model.layers:
        is_built.append(layer.built)
        if hasattr(layer, "hgq"):  # Activation layers
            is_built.append(layer.hgq.built)
        if hasattr(layer, "hgq_weight"):  # Compression layers
            is_built.append(layer.hgq_weight.built)
        if hasattr(layer, "hgq_bias"):  # Compression layers
            is_built.append(layer.hgq_bias.built)
    assert all(is_built)
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=True)(inputs)
    act = Activation("tanh")(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_conv2d_hgq")
    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)

    is_built = []
    for layer in model.layers:
        is_built.append(layer.built)
        if hasattr(layer, "hgq"):  # Activation layers
            is_built.append(layer.hgq.built)
        if hasattr(layer, "hgq_weight"):  # Compression layers
            is_built.append(layer.hgq_weight.built)
        if hasattr(layer, "hgq_bias"):  # Compression layers
            is_built.append(layer.hgq_bias.built)
    assert all(is_built)


# Activation Pruning


def test_ap_conv2d_channels_last_transpose(config_ap, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers_tf(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_ap, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_ap_conv1d_channels_last_transpose(config_ap, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers_tf(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers_tf(model_cl, config_ap, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_ap_depthwiseconv2d_channels_last_transpose(config_ap, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers_tf(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_ap, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_ap_dense_channels_last_transpose(config_ap, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers_tf(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers_tf(model_cl, config_ap, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


# Wanda


def test_wanda_conv2d_channels_last_transpose(config_wanda, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers_tf(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_wanda, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_wanda_conv1d_channels_last_transpose(config_wanda, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers_tf(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers_tf(model_cl, config_wanda, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_wanda_depthwiseconv2d_channels_last_transpose(config_wanda, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers_tf(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_wanda, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_wanda_dense_channels_last_transpose(config_wanda, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers_tf(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers_tf(model_cl, config_wanda, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


# PDP


def test_pdp_conv2d_channels_last_transpose(config_pdp, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers_tf(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_pdp, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_pdp_conv1d_channels_last_transpose(config_pdp, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers_tf(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers_tf(model_cl, config_pdp, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_pdp_depthwiseconv2d_channels_last_transpose(config_pdp, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers_tf(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_pdp, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_pdp_dense_channels_last_transpose(config_pdp, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)
    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers_tf(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers_tf(model_cl, config_pdp, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)
    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)

    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))
    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


# CS


def test_cs_conv2d_channels_last_transpose(config_cs, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers_tf(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1].weight
    s = model_cf.layers[1].pruning_layer.s.value
    new_s = np.zeros_like(s) + 0.1
    new_s = np.reshape(new_s, -1)
    new_s[: ops.size(s) // 2] = -1.0
    new_s = ops.reshape(new_s, s.shape)
    model_cf.layers[1].pruning_layer.s.assign(new_s)

    post_pretrain_functions(model_cf, config_cs)
    out_cf = model_cf(inp, training=True)
    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_cs, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    model_cl.layers[1].pruning_layer.s.assign(new_s)

    post_pretrain_functions(model_cl, config_cs)

    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.get_hard_mask(None)

    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.get_hard_mask(None)
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))
    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_cs_conv1d_channels_last_transpose(config_cs, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers_tf(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_cs)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers_tf(model_cl, config_cs, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_cs)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_cs_depthwiseconv2d_channels_last_transpose(config_cs, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers_tf(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_cs)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers_tf(model_cl, config_cs, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_cs)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_cs_dense_channels_last_transpose(config_cs, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers_tf(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1].weight

    post_pretrain_functions(model_cf, config_cs)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers_tf(model_cl, config_cs, inp.shape)
    model_cl.layers[1].weight.assign(weight_cf)
    post_pretrain_functions(model_cl, config_cs)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].weight, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].weight, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))
    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_calculate_pruning_budget(config_wanda, dense_input):
    sparsity = 0.75
    config_wanda["pruning_parameters"]["calculate_pruning_budget"] = True
    config_wanda["pruning_parameters"]["sparsity"] = sparsity

    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    out2 = Dense(OUT_FEATURES, use_bias=False)(out)
    model = keras.Model(inputs=inputs, outputs=out2, name="test_conv2d")

    # First layer will have 50% sparsity
    weight = np.ones(IN_FEATURES * OUT_FEATURES).astype(np.float32)
    weight[: IN_FEATURES * OUT_FEATURES // 2] = 0.001
    weight = ops.reshape(ops.convert_to_tensor(weight), (IN_FEATURES, OUT_FEATURES))
    weight2 = ops.reshape(ops.linspace(0.01, 0.99, OUT_FEATURES * OUT_FEATURES), (OUT_FEATURES, OUT_FEATURES))

    model = add_compression_layers_tf(model, config_wanda, dense_input.shape)
    model.layers[1].weight.assign(weight)
    model.layers[2].weight.assign(weight2)
    # Triggers calculation of pruning budget for PDP and Wanda
    post_pretrain_functions(model, config_wanda)
    total_weights = IN_FEATURES * OUT_FEATURES + OUT_FEATURES * OUT_FEATURES
    remaining_weights = 0
    for layer in model.layers:
        if hasattr(layer, "pruning_layer"):
            calculated_sparsity = layer.pruning_layer.sparsity
            remaining_weights += (1 - calculated_sparsity) * ops.cast(ops.size(layer.weight), "float32")
    # First layer should have 50% sparsity, total sparsity should be around 75%
    assert model.layers[1].pruning_layer.sparsity == 0.5
    np.testing.assert_allclose(remaining_weights / total_weights, 1 - sparsity, atol=1e-3, rtol=0)


def test_trigger_post_pretraining(config_pdp, conv2d_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    act1 = Activation("tanh")(out)
    out2 = Dense(OUT_FEATURES, use_bias=False)(act1)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers_tf(model, config_pdp, conv2d_input.shape)

    assert model.layers[1].pruning_layer.is_pretraining is True
    assert model.layers[2].is_pretraining is True
    assert model.layers[3].pruning_layer.is_pretraining is True
    assert model.layers[4].is_pretraining is True

    post_pretrain_functions(model, config_pdp)

    assert model.layers[1].pruning_layer.is_pretraining is False
    assert model.layers[2].is_pretraining is False
    assert model.layers[3].pruning_layer.is_pretraining is False
    assert model.layers[4].is_pretraining is False


def test_hgq_weight_shape(config_pdp, dense_input):
    config_pdp["quantization_parameters"]["enable_quantization"] = True
    config_pdp["quantization_parameters"]["use_high_granularity_quantization"] = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    act1 = Activation("tanh")(out)
    out2 = Dense(OUT_FEATURES, use_bias=False)(act1)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)
    assert model.layers[1].hgq_weight.quantizer._i.shape == model.layers[1].weight.shape
    layer_2_input_shape = [1] + list(model.layers[2].input.shape[1:])
    assert model.layers[2].hgq.quantizer._i.shape == layer_2_input_shape

    config_pdp["quantization_parameters"]["hgq_heterogeneous"] = False
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    act1 = Activation("tanh")(out)
    out2 = Dense(OUT_FEATURES, use_bias=False)(act1)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers_tf(model, config_pdp, dense_input.shape)
    assert model.layers[1].hgq_weight.quantizer._i.shape == (1, 1)
    assert model.layers[2].hgq.quantizer._i.shape == (1, 1)
