import pytest
from keras import ops
from keras.random import shuffle

from pquant.pruning_methods.pdp import PDP


@pytest.fixture
def config():
    return {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "epsilon": 1.0,
            "sparsity": 0.75,
            "temperature": 1e-5,
            "threshold_decay": 0.0,
            "structured_pruning": False,
        }
    }


IN_FEATURES = 8
OUT_FEATURES = 16
KERNEL_SIZE = 3


def test_linear_unstructured(config):
    sparsity = config["pruning_parameters"]["sparsity"]

    inp = ops.linspace(-1, 1, num=OUT_FEATURES * IN_FEATURES)
    threshold_point = int(OUT_FEATURES * IN_FEATURES * sparsity) - 1
    threshold_value = sorted(ops.abs(inp))[threshold_point]
    inp = shuffle(inp)
    mask = ops.cast((ops.abs(inp) > threshold_value), inp.dtype)
    inp = ops.reshape(inp, (OUT_FEATURES, IN_FEATURES))
    mask = ops.reshape(mask, inp.shape)

    pdp = PDP(config, "linear")
    pdp.build(inp.shape)
    pdp.post_pre_train_function()
    result = pdp(inp)

    result = ops.cast((result != 0.0), result.dtype)

    assert ops.all(ops.equal(result, mask))
    expected_nonzero_count = ops.convert_to_tensor(OUT_FEATURES * IN_FEATURES * (1.0 - sparsity))
    assert ops.all(ops.equal(expected_nonzero_count, ops.count_nonzero(result)))


def test_conv_unstructured(config):
    size = IN_FEATURES * OUT_FEATURES * KERNEL_SIZE * KERNEL_SIZE
    sparsity = config["pruning_parameters"]["sparsity"]

    inp = ops.linspace(-1, 1, num=size)
    threshold_point = int(OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE * sparsity) - 1
    threshold_value = sorted(ops.abs(inp))[threshold_point]
    inp = shuffle(inp)
    mask = ops.cast((ops.abs(inp) > threshold_value), inp.dtype)
    pdp = PDP(config, "conv")
    pdp.post_pre_train_function()
    inp = ops.reshape(inp, (IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, KERNEL_SIZE))
    mask = ops.reshape(mask, inp.shape)
    pdp.build(inp.shape)

    result = pdp(inp)
    result = ops.cast((result != 0.0), result.dtype)
    assert ops.all(ops.equal(result, mask))
    expected_nonzero_count = ops.convert_to_tensor(OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE * (1.0 - sparsity))
    assert ops.all(ops.equal(expected_nonzero_count, ops.count_nonzero(result)))


def test_linear_structured(config):
    sparsity = config["pruning_parameters"]["sparsity"]
    config["pruning_parameters"]["structured_pruning"] = True

    inp = ops.linspace(-1, 1, num=OUT_FEATURES)
    threshold_point = int(OUT_FEATURES * sparsity) - 1
    threshold_value = sorted(ops.abs(inp))[threshold_point]
    inp = shuffle(inp)
    mask = ops.cast((ops.abs(inp) > threshold_value), inp.dtype)
    inp = ops.tile(inp, (IN_FEATURES, 1))
    mask = ops.tile(mask, (IN_FEATURES, 1))
    threshold_point = int(OUT_FEATURES * sparsity)
    # In the matrix of shape NxM, prune over the M dimension
    pdp = PDP(config, "linear")
    pdp.post_pre_train_function()
    pdp.build(inp.shape)

    result = pdp(inp)
    result = ops.cast((result != 0.0), result.dtype)
    assert ops.all(ops.equal(result, mask))
    expected_nonzero_count = ops.convert_to_tensor(OUT_FEATURES * IN_FEATURES * (1.0 - sparsity))
    assert ops.all(ops.equal(expected_nonzero_count, ops.count_nonzero(result)))


def test_conv_structured(config):
    config["pruning_parameters"]["structured_pruning"] = True
    sparsity = config["pruning_parameters"]["sparsity"]

    inp = ops.ones(shape=(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, KERNEL_SIZE))
    channel_dim = ops.linspace(-1, 1, num=OUT_FEATURES)
    threshold_point = int(OUT_FEATURES * sparsity) - 1
    threshold_value = sorted(ops.abs(channel_dim))[threshold_point]
    channel_dim = shuffle(channel_dim)
    mask = ops.cast((ops.abs(channel_dim) > threshold_value), inp.dtype)
    mask = ops.expand_dims(ops.expand_dims(ops.expand_dims(mask, axis=0), axis=-1), axis=-1)
    mask = ops.tile(mask, (IN_FEATURES, 1, KERNEL_SIZE, KERNEL_SIZE))

    mult = ops.expand_dims(ops.expand_dims(ops.expand_dims(channel_dim, axis=0), axis=-1), axis=-1)
    inp *= mult

    pdp = PDP(config, "conv")
    pdp.post_pre_train_function()
    pdp.build(inp.shape)
    result = pdp(inp)

    result = ops.cast((result != 0.0), result.dtype)
    assert ops.all(ops.equal(result, mask))
    expected_nonzero_count = ops.convert_to_tensor(OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE * (1.0 - sparsity))
    assert ops.all(ops.equal(expected_nonzero_count, ops.count_nonzero(result)))
