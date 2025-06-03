import keras


def add_default_layer_quantization_pruning_to_config(model, config):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            add_default_layer_quantization_pruning_to_config_torch,
        )

        return add_default_layer_quantization_pruning_to_config_torch(config, model)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import (
            add_default_layer_quantization_pruning_to_config_tf,
        )

        return add_default_layer_quantization_pruning_to_config_tf(config, model)


def add_compression_layers(model, config, input_shape):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            add_compression_layers_torch,
        )

        return add_compression_layers_torch(model, config, input_shape)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import add_compression_layers_tf

        return add_compression_layers_tf(model, config, input_shape)


def get_layer_keep_ratio(model):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            get_layer_keep_ratio_torch,
        )

        return get_layer_keep_ratio_torch(model)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import get_layer_keep_ratio_tf

        return get_layer_keep_ratio_tf(model)


def get_model_losses(model, losses):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            get_model_losses_torch,
        )

        return get_model_losses_torch(model, losses)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import get_model_losses_tf

        return get_model_losses_tf(model, losses)


def remove_pruning_from_model(model, config):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            remove_pruning_from_model_torch,
        )

        return remove_pruning_from_model_torch(model, config)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import (
            remove_pruning_from_model_tf,
        )

        return remove_pruning_from_model_tf(model, config)
