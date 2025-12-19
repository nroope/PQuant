# @Author: Arghya Ranjan Das
# file: src/pquant/pruning_methods/mdmm.py
# modified by:


import inspect

import keras
from keras import ops

from pquant.core.constants import CONSTRAINT_REGISTRY, METRIC_REGISTRY

# -------------------------------------------------------------------
#                   MDMM Layer
# -------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="PQuant")
class MDMM(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.config = config
        self.layer_type = layer_type
        self.constraint_layer = None
        self.penalty_loss = None
        self.built = False
        self.is_finetuning = False

    def build(self, input_shape):
        pruning_parameters = self.config.pruning_parameters
        metric_type = pruning_parameters.metric_type
        constraint_type = pruning_parameters.constraint_type
        target_value = pruning_parameters.target_value
        target_sparsity = pruning_parameters.target_sparsity
        l0_mode = pruning_parameters.l0_mode
        scale_mode = pruning_parameters.scale_mode

        candidate_kwargs = {
            "epsilon": pruning_parameters.epsilon,
            "target_sparsity": target_sparsity,
            "l0_mode": l0_mode,
            "scale_mode": scale_mode,
            "rf": pruning_parameters.rf,
        }

        metric_cls = METRIC_REGISTRY.get(metric_type)
        sig = inspect.signature(getattr(metric_cls, "__init__", metric_cls))
        metric_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None and k in sig.parameters}
        if metric_cls:
            metric_fn = metric_cls(**metric_kwargs)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        common_args = {
            "metric_fn": metric_fn,
            "target_value": target_value,
            "scale": self.config.pruning_parameters.scale,
            "damping": self.config.pruning_parameters.damping,
            "use_grad": self.config.pruning_parameters.use_grad,
            "lr": self.config.training_parameters.lr,
        }

        constraint_type_cls = CONSTRAINT_REGISTRY.get(constraint_type)
        if constraint_type_cls:
            self.constraint_layer = constraint_type_cls(**common_args)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        self.mask = ops.ones(input_shape)
        self.constraint_layer.build(input_shape)
        super().build(input_shape)
        self.built = True

    def call(self, weight):
        if not self.built:
            self.build(weight.shape)

        if self.is_finetuning:
            self.penalty_loss = 0.0
            weight = weight * self.get_hard_mask(weight)
        else:
            self.penalty_loss = self.constraint_layer(weight)
        epsilon = self.config.pruning_parameters.epsilon
        self.hard_mask = ops.cast(ops.abs(weight) > epsilon, weight.dtype)
        return weight

    def get_hard_mask(self, weight=None):
        if weight is None:
            return self.hard_mask
        epsilon = self.config.pruning_parameters.epsilon
        return ops.cast(ops.abs(weight) > epsilon, weight.dtype)

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask(weight)) / ops.size(weight)  # Should this be subtracted from 1.0?

    def calculate_additional_loss(self):
        if self.penalty_loss is None:
            raise ValueError("Penalty loss has not been calculated. Call the layer with weights first.")
        else:
            penalty_loss = ops.sum(self.penalty_loss)

        return penalty_loss

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def pre_finetune_function(self):
        # Freeze the weights
        # Set lmbda(s) to zero
        self.is_finetuning = True
        if hasattr(self.constraint_layer, 'module'):
            self.constraint_layer.module.turn_off()
        else:
            self.constraint_layer.turn_off()

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def post_pre_train_function(self):
        pass

    def post_round_function(self):
        pass

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "config": self.config.get_dict(),
                "layer_type": self.layer_type,
            }
        )
        return config
