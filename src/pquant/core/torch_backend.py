import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from pquant.core.activations_quantizer import (
    QuantizedPooling,
    QuantizedReLU,
    QuantizedTanh,
)
from pquant.core.backend_interface import BackendInterface
from pquant.core.compressed_layers import (
    CompressedLayerBase,
    CompressedLayerConv1d,
    CompressedLayerConv2d,
    CompressedLayerLinear,
)


class TorchBackend(BackendInterface):
    def iterative_train(self, model, config, train_func, valid_func, **kwargs):
        """
        Generic training loop, user provides training and validation functions
        """
        epoch = torch.tensor(0)  # Keeps track of all the epochs completed
        training_config = config.training_parameters
        if training_config.pretraining_epochs > 0:
            for e in range(training_config.pretraining_epochs):
                model.train()
                self.pre_epoch_functions(model, e, training_config.pretraining_epochs)
                train_func(model, epoch=epoch, **kwargs)
                model.eval()
                valid_func(model, epoch=epoch, **kwargs)
                self.post_epoch_functions(model, e, training_config.pretraining_epochs)
                epoch += 1
        self.post_pretrain_functions(model, config)
        for r in range(training_config.rounds):
            for e in range(training_config.epochs):
                model.train()
                if r == 0 and training_config.save_weights_epoch == e:
                    self.save_weights_functions(model)
                self.pre_epoch_functions(model, e, training_config.epochs)
                train_func(model, epoch=epoch, **kwargs)
                model.eval()
                valid_func(model, epoch=epoch, **kwargs)
                self.post_epoch_functions(model, e, training_config.epochs)
                epoch += 1
            self.call_post_round_functions(model, training_config.rewind, training_config.rounds, r)
        self.pre_finetune_functions(model)
        if training_config.fine_tuning_epochs > 0:
            for e in range(training_config.fine_tuning_epochs):
                model.train()
                self.pre_epoch_functions(model, e, training_config.fine_tuning_epochs)
                train_func(model, epoch=epoch, **kwargs)
                model.eval()
                valid_func(model, epoch=epoch, **kwargs)
                self.post_epoch_functions(model, e, training_config.fine_tuning_epochs)
                epoch += 1
        return model

    def create_default_layer_quantization_pruning_config(self, model):
        config = {"layer_specific": {}, "disable_pruning_for_layers": []}
        for name, layer in model.named_modules():
            if layer.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d]:
                if layer.bias is None:
                    config["layer_specific"][name] = {"weight": {"integer_bits": 0, "fractional_bits": 7}}
                else:
                    config["layer_specific"][name] = {
                        "weight": {"integer_bits": 0, "fractional_bits": 7},
                        "bias": {"integer_bits": 0, "fractional_bits": 7},
                    }
                config["disable_pruning_for_layers"].append(name)
            elif layer.__class__ in [nn.Tanh, nn.ReLU, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
                config["layer_specific"][name] = {"integer_bits": 0, "fractional_bits": 7}
        return config

    def add_default_layer_quantization_pruning_to_config(self, model, config):
        custom_scheme = self.create_default_layer_quantization_pruning_config(model)
        config.quantization_parameters.layer_specific = custom_scheme["layer_specific"]
        config.pruning_parameters.disable_pruning_for_layers = custom_scheme["disable_pruning_for_layers"]
        return config

    def remove_pruning_from_model(self, module, config):
        for name, layer in module.named_children():
            if isinstance(layer, CompressedLayerLinear):
                if config.pruning_parameters.pruning_method == "pdp":  # Find better solution later
                    if config.training_parameters.pruning_first:
                        weight = layer.weight
                        if layer.enable_pruning:
                            weight = layer.pruning_layer.get_hard_mask(weight) * weight
                        weight, bias = layer.quantize(weight, layer.bias)
                    else:
                        weight, bias = layer.quantize(layer.weight, layer.bias)
                        if layer.enable_pruning:
                            weight = layer.pruning_layer.get_hard_mask(weight) * weight
                else:
                    weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
                out_features = layer.out_features
                bias_values = bias
                in_features = layer.in_features
                bias = True if bias_values is not None else False
                setattr(module, name, nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
                getattr(module, name).weight.data.copy_(weight)
                if getattr(module, name).bias is not None:
                    getattr(module, name).bias.data.copy_(bias_values.data)
            elif isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d)):
                if config.pruning_parameters.pruning_method == "pdp":  # Find better solution later
                    if config.training_parameters.pruning_first:
                        weight = layer.weight
                        if layer.enable_pruning:
                            weight = layer.pruning_layer.get_hard_mask(weight) * weight
                        weight, bias = layer.quantize(weight, layer.bias)
                    else:
                        weight, bias = layer.quantize(layer.weight, layer.bias)
                        if layer.enable_pruning:
                            weight = layer.pruning_layer.get_hard_mask(weight) * weight
                else:
                    weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
                bias_values = bias
                bias = True if bias_values is not None else False
                conv = nn.Conv2d if isinstance(layer, CompressedLayerConv2d) else nn.Conv1d
                setattr(
                    module,
                    name,
                    conv(
                        layer.in_channels,
                        layer.out_channels,
                        layer.kernel_size,
                        layer.stride,
                        layer.padding,
                        layer.dilation,
                        layer.groups,
                        bias,
                        layer.padding_mode,
                    ),
                )
                getattr(module, name).weight.data.copy_(weight)
                if getattr(module, name).bias is not None:
                    getattr(module, name).bias.data.copy_(bias_values.data)
            else:
                self.remove_pruning_from_model(layer, config)
        return module

    def add_quantized_activations_to_model_layer(self, module, config):
        if not config.quantization_parameters.enable_quantization:
            return module
        # Replaces ReLU and Tanh layers with quantized versions
        for name, layer in module.named_children():
            i = config.quantization_parameters.default_integer_bits
            f = config.quantization_parameters.default_fractional_bits
            if layer.__class__ in [nn.ReLU]:
                # For ReLU, if using default values, add 1 bit since values are unsigned.
                # Otherwise user provides bits. TODO: Find better way to do this
                f = config.quantization_parameters.default_fractional_bits + 1
                relu = QuantizedReLU(config, i=i, f=f)
                setattr(module, name, relu)
            elif layer.__class__ in [nn.Tanh]:
                tanh = QuantizedTanh(config, i=0.0, f=f)
                setattr(module, name, tanh)
            elif layer.__class__ in [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
                new_layer = QuantizedPooling(config, layer)
                setattr(module, name, new_layer)
            else:
                layer = self.add_quantized_activations_to_model_layer(layer, config)
        return module

    def add_pruning_to_model(self, module, config):
        for name, layer in module.named_children():
            if layer.__class__ is nn.Linear:
                sparse_layer = CompressedLayerLinear(config, layer, "linear")
                sparse_layer.pruning_layer.build(layer.weight.shape)
                setattr(module, name, sparse_layer)
            elif layer.__class__ is nn.Conv2d:
                sparse_layer = CompressedLayerConv2d(config, layer, "conv")
                sparse_layer.pruning_layer.build(layer.weight.shape)
                setattr(module, name, sparse_layer)
            elif layer.__class__ is nn.Conv1d:
                sparse_layer = CompressedLayerConv1d(config, layer, "conv")
                sparse_layer.pruning_layer.build(layer.weight.shape)
                setattr(module, name, sparse_layer)
            else:
                self.add_pruning_to_model(layer, config)
        return module

    def disable_pruning_from_layers(self, module, config):
        for name, layer in module.named_modules():
            enable_pruning = name not in config.pruning_parameters.disable_pruning_for_layers
            if (
                layer.__class__ in [CompressedLayerLinear, CompressedLayerConv2d, CompressedLayerConv1d]
                and not enable_pruning
            ):
                layer.enable_pruning = enable_pruning
        return module

    def add_layer_specific_quantization_to_model(self, module, config):
        for name, layer in module.named_modules():
            if isinstance(layer, CompressedLayerBase):
                if name in config.quantization_parameters.layer_specific:
                    if "weight" in config.quantization_parameters.layer_specific[name]:
                        weight_int_bits = config.quantization_parameters.layer_specific[name]["weight"]["integer_bits"]
                        weight_fractional_bits = config.quantization_parameters.layer_specific[name]["weight"][
                            "fractional_bits"
                        ]
                        layer.i_weight = torch.tensor(weight_int_bits)
                        layer.f_weight = torch.tensor(weight_fractional_bits)
                    if "bias" in config.quantization_parameters.layer_specific[name]:
                        bias_int_bits = config.quantization_parameters.layer_specific[name]["bias"]["integer_bits"]
                        bias_fractional_bits = config.quantization_parameters.layer_specific[name]["bias"]["fractional_bits"]
                        layer.i_bias = torch.tensor(bias_int_bits)
                        layer.f_bias = torch.tensor(bias_fractional_bits)
                layer.build(None)
            elif layer.__class__ in [QuantizedPooling, QuantizedReLU, QuantizedTanh]:
                if name in config.quantization_parameters.layer_specific:
                    i = config.quantization_parameters.layer_specific[name]["integer_bits"]
                    f = config.quantization_parameters.layer_specific[name]["fractional_bits"]
                    layer.set_activation_bits(i, f)
        return module

    def add_quantized_activations_to_model_functional(self, module, config):
        # Currently not in use. TODO: Fix this
        if config.quantization_parameters.use_high_granularity_quantization:
            return module
        # Replaces functional activation calls with quantized versions
        traced_model = symbolic_trace(module)
        for node in traced_model.graph.nodes:
            if node.op in ["call_method", "call_function"] and (
                node.target == "tanh" or "function relu" in str(node.target)
            ):
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

    def add_compression_layers(self, model, config, input_shape):
        model = self.add_quantized_activations_to_model_layer(model, config)
        # model = self.add_quantized_activations_to_model_functional(model, config)
        model = self.add_pruning_to_model(model, config)
        model = self.disable_pruning_from_layers(model, config)
        model = self.add_layer_specific_quantization_to_model(model, config)
        model(torch.rand(input_shape, device=next(model.parameters()).device))
        return model

    def post_epoch_functions(self, model, epoch, total_epochs, **kwargs):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)

    def pdp_setup(self, model, config):
        """
        Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
        how large percentage of weights in the layer is smaller than the global threshold
        """
        global_weights = None
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                if global_weights is None:
                    global_weights = layer.weight.flatten()
                else:
                    global_weights = torch.concat((global_weights, layer.weight.flatten()))

        abs_global_weights = torch.abs(global_weights)
        global_weight_topk, _ = torch.topk(abs_global_weights, abs_global_weights.numel())
        threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * global_weight_topk.numel())]
        global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
        idx = 0
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                weight_size = layer.weight.numel()
                w = torch.sum(global_weights_below_threshold[idx : idx + weight_size])
                layer.pruning_layer.init_r = w / weight_size
                layer.pruning_layer.sparsity = w / weight_size  # Wanda
                idx += weight_size

    def post_pretrain_functions(self, model, config):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.pruning_layer.post_pre_train_function()
            elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
                layer.post_pre_train_function()
        if config.pruning_parameters.pruning_method == "pdp" or (
            config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
        ):
            self.pdp_setup(model, config)

    def pre_epoch_functions(self, model, epoch, total_epochs):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.pruning_layer.pre_epoch_function(epoch, total_epochs)

    def pre_finetune_functions(self, model):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.pruning_layer.pre_finetune_function()

    def save_weights_functions(self, model):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.save_weights()

    def post_round_functions(self, model):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.pruning_layer.post_round_function()

    def rewind_weights_functions(self, model):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                layer.rewind_weights()

    @torch.no_grad
    def get_layer_keep_ratio(self, model):
        total_w = 0
        remaining_weights = 0
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                if layer.pruning_first:
                    weight = layer.weight
                    if layer.enable_pruning:
                        weight = layer.pruning_layer.get_hard_mask(weight) * weight
                    weight, bias = layer.quantize(weight, layer.bias)
                    total_w += weight.numel()
                    rem = torch.count_nonzero(weight)
                    remaining_weights += rem
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    if layer.enable_pruning:
                        weight = layer.pruning_layer.get_hard_mask(weight) * weight
                    total_w += weight.numel()
                    rem = torch.count_nonzero(weight)
                    remaining_weights += rem
            elif layer.__class__ in (nn.Conv2d, nn.Conv1d, nn.Linear):
                total_w += layer.weight.numel()
                remaining_weights += torch.count_nonzero(layer.weight)
        if total_w != 0:
            return remaining_weights / total_w
        return 0.0

    def get_model_losses(self, model, losses):
        for layer in model.modules():
            if isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d, CompressedLayerLinear)):
                loss = layer.pruning_layer.calculate_additional_loss()
                if layer.use_high_granularity_quantization:
                    loss += layer.hgq_loss()
                losses += loss
            elif isinstance(layer, (QuantizedReLU, QuantizedTanh, QuantizedPooling)):
                if layer.use_high_granularity_quantization:
                    losses += layer.hgq_loss()
        return losses
