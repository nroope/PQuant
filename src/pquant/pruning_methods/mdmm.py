# @Author: Arghya Ranjan Das
# file: src/pquant/pruning_methods/mdmm.py
# modified by:


import abc

import keras
from keras import ops


@ops.custom_gradient
def flip_gradient(x, scale=-1.0):
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        scale_t = ops.convert_to_tensor(scale, dtype=upstream.dtype)
        return (ops.multiply(upstream, scale_t),)  # ops.abs()

    return x, grad


# Abstract base class for constraints
@keras.utils.register_keras_serializable(name="Constraint")
class Constraint(keras.layers.Layer):
    def __init__(self, lmbda_init=1.0, scale=1.0, damping=1.0, **kwargs):
        self.use_grad_ = bool(kwargs.pop("use_grad", True))
        self.lr_ = float(kwargs.pop("lr", 0.0))
        super().__init__(**kwargs)

        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(scale, dtype=dtype),
            trainable=False,
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(damping, dtype=dtype),
            trainable=False,
        )
        self.lmbda = self.add_weight(
            name=f'{self.name}_lmbda',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(lmbda_init, dtype=dtype),
            trainable=self.use_grad_,
        )

        if not self.use_grad_:
            self.prev_infs = self.add_weight(
                name=f'{self.name}_prev_infs',
                shape=(),
                initializer=lambda shape, dtype: ops.convert_to_tensor(0.0, dtype=dtype),
                trainable=False,
            )

    def call(self, weight):
        """Calculates the penalty from a given infeasibility measure."""
        raw_infeasibility = self.get_infeasibility(weight)
        infeasibility = self.pipe_infeasibility(raw_infeasibility)

        if self.use_grad_:
            ascent_lmbda = flip_gradient(self.lmbda)
            # ascent_lmbda = ops.maximum(ascent_lmbda, 0.0)
        else:
            lmbda_step = self.lr_ * self.scale * self.prev_infs
            ascent_lmbda = self.lmbda + lmbda_step
            self.lmbda.assign_add(lmbda_step)
            self.prev_infs.assign(infeasibility)

        l_term = ascent_lmbda * infeasibility
        damp_term = self.damping * ops.square(infeasibility) / 2
        penalty = self.scale * (l_term + damp_term)

        return penalty

    @abc.abstractmethod
    def get_infeasibility(self, weight):
        """Must be implemented by subclasses to define the violation."""
        raise NotImplementedError

    def pipe_infeasibility(self, infeasibility):
        """Optional transformation of raw infeasibility.
        Default is identity. Subclasses may override."""
        return infeasibility

    def turn_off(self):
        if not self.use_grad_:
            self.lr_ = 0.0
        self.scale.assign(0.0)
        self.lmbda.assign(0.0)


# -------------------------------------------------------------------
#               Generic Constraint Classes
# -------------------------------------------------------------------


@keras.utils.register_keras_serializable(name="EqualityConstraint")
class EqualityConstraint(Constraint):
    """Constraint for g(w) == target_value."""

    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value

    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.abs(infeasibility)


@keras.utils.register_keras_serializable(name="LessThanOrEqualConstraint")
class LessThanOrEqualConstraint(Constraint):
    """Constraint for g(w) <= target_value."""

    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value

    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.maximum(infeasibility, 0.0)


@keras.utils.register_keras_serializable(name="GreaterThanOrEqualConstraint")
class GreaterThanOrEqualConstraint(Constraint):
    """Constraint for g(w) >= target_value."""

    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value

    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = self.target_value - metric_value
        return ops.maximum(infeasibility, 0.0)


# -------------------------------------------------------------------
#                   Metric Functions
# -------------------------------------------------------------------


class UnstructuredSparsityMetric:
    """L0-L1 based metric"""

    """Calculates the ratio of non-zero weights in a tensor."""

    def __init__(self, l0_mode='coarse', scale_mode="mean", epsilon=1e-3, target_sparsity=0.8, alpha=100.0):
        # Note: scale_mode:"sum" give very high losses for large model
        assert l0_mode in ['coarse', 'smooth'], "Mode must be 'coarse' or 'smooth'"
        assert scale_mode in ['sum', 'mean'], "Scale mode must be 'sum' or 'mean'"
        assert 0 <= target_sparsity <= 1, "target_sparsity must be between 0 and 1"
        self.l0_mode = l0_mode
        self.scale_mode = scale_mode
        self.target_sparsity = float(target_sparsity)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)

        self.l0_fn = None
        self._scaling = None

        self.build()

    def build(self):
        # l0 term -> number of zero weights/number of weights
        if self.l0_mode == 'coarse':
            self.l0_fn = self._coarse_l0
        elif self.l0_mode == 'smooth':
            self.l0_fn = self._smooth_l0

        if self.scale_mode == 'mean':
            self._scaling = self._mean_scaling
        elif self.scale_mode == 'sum':
            self._scaling = self._sum_scaling

    def _sum_scaling(self, fn_value, num):
        return fn_value

    def _mean_scaling(self, fn_value, num):
        return fn_value / num

    def _coarse_l0(self, weight_vector):
        return ops.mean(ops.cast(ops.abs(weight_vector) <= self.epsilon, "float32"))

    def _smooth_l0(self, weight_vector):
        """Differentiable approximation of L0 norm using Keras ops."""
        return ops.mean(ops.exp(-self.alpha * ops.square(weight_vector)))

    def __call__(self, weight):
        num_weights = ops.cast(ops.size(weight), weight.dtype)
        weights_vector = ops.reshape(weight, [-1])

        l0_term = self.l0_fn(weights_vector)
        l1_term = ops.sum(ops.abs(weights_vector))

        # farctor by constrction goes to zero when l0_term == target_sparsiity
        factor = ops.square(self.target_sparsity) - ops.square(l0_term)
        fn_value = factor * l1_term
        fn_value = self._scaling(fn_value, num_weights)

        return fn_value


class StructuredSparsityMetric:
    """Calculates the ratio of near-zero weight groups (based on Reuse Factor: rf)."""

    def __init__(self, rf=1, epsilon=1e-3):
        self.rf = rf
        self.epsilon = epsilon

    def __call__(self, weight):
        original_shape = weight.shape
        w_reshaped = ops.reshape(weight, (original_shape[0], -1))
        num_weights = ops.shape(w_reshaped)[1]

        padding = (self.rf - num_weights % self.rf) % self.rf
        w_padded = ops.pad(w_reshaped, [[0, 0], [0, padding]])

        groups = ops.reshape(w_padded, (original_shape[0], -1, self.rf))
        group_norms = ops.sqrt(ops.sum(ops.square(groups), axis=-1))
        zero_groups = ops.less_equal(group_norms, self.epsilon)
        num_groups = ops.cast(ops.size(group_norms), "float32")

        return ops.sum(ops.cast(zero_groups, "float32")) / num_groups


# -------------------------------------------------------------------
#                   MDMM Layer
# -------------------------------------------------------------------


class MDMM(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layer_type = layer_type
        self.constraint_layer = None
        self.penalty_loss = None
        self.built = False
        self.is_finetuning = False

    def build(self, input_shape):
        metric_type = self.config["pruning_parameters"].get("metric_type", "UnstructuredSparsity")
        constraint_type = self.config["pruning_parameters"].get("constraint_type", "GreaterThanOrEqual")
        target_value = self.config["pruning_parameters"].get("target_value", 0.0)
        target_sparsity = self.config["pruning_parameters"].get("target_sparsity", 0.9)
        l0_mode = self.config["pruning_parameters"].get("l0_mode", "coarse")
        scale_mode = self.config["pruning_parameters"].get("scale_mode", "mean")

        if metric_type == "UnstructuredSparsity":
            metric_fn = UnstructuredSparsityMetric(
                epsilon=self.config["pruning_parameters"].get("epsilon", 1e-5),
                target_sparsity=target_sparsity,
                l0_mode=l0_mode,
                scale_mode=scale_mode,
            )
        elif metric_type == "StructuredSparsity":
            metric_fn = StructuredSparsityMetric(
                rf=self.config["rf"], epsilon=self.config["pruning_parameters"].get("epsilon", 1e-5)
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        common_args = {
            "metric_fn": metric_fn,
            "target_value": target_value,
            "scale": self.config["pruning_parameters"].get("scale", 1.0),
            "damping": self.config["pruning_parameters"].get("damping", 1.0),
            "use_grad": self.config["pruning_parameters"].get("use_grad", True),
            "lr": self.config.get("lr", 0.0),
        }

        if constraint_type == "Equality":
            self.constraint_layer = EqualityConstraint(**common_args)
        elif constraint_type == "LessThanOrEqual":
            self.constraint_layer = LessThanOrEqualConstraint(**common_args)
        elif constraint_type == "GreaterThanOrEqual":
            self.constraint_layer = GreaterThanOrEqualConstraint(**common_args)
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

        return weight

    def get_hard_mask(self, weight):
        epsilon = self.config["pruning_parameters"].get("epsilon", 1e-5)
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
