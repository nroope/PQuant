import keras


def add_default_layer_quantization_pruning_to_config(model, config):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            add_default_layer_quantization_pruning_to_config_torch,
        )

        return add_default_layer_quantization_pruning_to_config_torch(model, config)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import (
            add_default_layer_quantization_pruning_to_config_tf,
        )

        return add_default_layer_quantization_pruning_to_config_tf(model, config)


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


def post_training_prune(model, calibration_data, config):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.compressed_layers_torch import (
            add_compression_layers_torch,
            post_pretrain_functions,
            remove_pruning_from_model_torch,
        )

        t_delta = config["pruning_parameters"]["t_delta"]
        config["pruning_parameters"]["t_start_collecting_batch"] = 0
        for i in range(t_delta):
            inputs = calibration_data[i]
            if i == 0:
                model = add_compression_layers_torch(model, config, inputs.shape)
                post_pretrain_functions(model, config)
            model(inputs)
        return remove_pruning_from_model_torch(model, config)
    else:
        from pquant.core.tf_impl.compressed_layers_tf import (
            add_compression_layers_tf,
            post_pretrain_functions,
            remove_pruning_from_model_tf,
        )

        t_delta = config["pruning_parameters"]["t_delta"]
        config["pruning_parameters"]["t_start_collecting_batch"] = 0

        for i in range(t_delta):
            inputs = calibration_data[i]
            if i == 0:
                model = add_compression_layers_tf(model, config, inputs.shape)
                post_pretrain_functions(model, config)
            model(inputs, training=True)  # True so pruning works
        return remove_pruning_from_model_tf(model, config)
