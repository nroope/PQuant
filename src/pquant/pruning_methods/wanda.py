import keras
from keras import ops


class Wanda(keras.layers.Layer):

    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.act_type = "relu"
        self.t = 0
        self.layer_type = layer_type
        self.batches_collected = 0
        self.inputs = None
        self.total = 0.0
        self.done = False
        self.sparsity = self.config["pruning_parameters"]["sparsity"]
        self.is_pretraining = True
        self.N = self.config["pruning_parameters"]["N"]
        self.M = self.config["pruning_parameters"]["M"]
        self.t_start_collecting_batch = self.config["pruning_parameters"]["t_start_collecting_batch"]

    def build(self, input_shape):
        self.mask = ops.ones(input_shape)
        super().build(input_shape)

    def get_mask(self, weight, metric, sparsity):
        d0, d1 = metric.shape
        keep_idxs = ops.argsort(metric, axis=1)[:, int(d1 * sparsity) :] + ops.arange(d0)[:, None] * d1
        keep_idxs = ops.ravel(keep_idxs)
        kept_values = ops.reshape(
            ops.scatter(keep_idxs[:, None], ops.take(ops.ravel(weight), keep_idxs), ops.array((ops.size(weight),))),
            weight.shape,
        )
        mask = ops.cast(kept_values != 0, weight.dtype)
        return mask

    def handle_linear(self, x, weight):
        norm = ops.norm(x, ord=2, axis=0)
        metric = ops.abs(weight) * norm
        if self.N is not None and self.M is not None:
            # N:M pruning
            metric_reshaped = ops.reshape(metric, (-1, self.M))
            weight_reshaped = ops.reshape(weight, (-1, self.M))
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.N / self.M)
            self.mask = ops.reshape(mask, weight.shape)
        else:
            # Unstructured pruning
            metric_reshaped = ops.reshape(metric, (1, -1))
            weight_reshaped = ops.reshape(weight, (1, -1))
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.sparsity)
            self.mask = ops.reshape(mask, weight.shape)

    def handle_conv(self, x, weight):
        inputs_avg = ops.mean(ops.reshape(x, (x.shape[0], x.shape[1], -1)), axis=0)
        norm = ops.norm(inputs_avg, ord=2, axis=-1)
        if len(weight.shape) == 3:
            norm = ops.reshape(norm, [1] + list(norm.shape) + [1])
        else:
            norm = ops.reshape(norm, [1] + list(norm.shape) + [1, 1])
        metric = ops.abs(weight) * norm
        if self.N is not None and self.M is not None:
            # N:M pruning
            metric_reshaped = ops.reshape(metric, (-1, self.M))
            weight_reshaped = ops.reshape(weight, (-1, self.M))
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.N / self.M)
            self.mask = ops.reshape(mask, weight.shape)
        else:
            # Unstructured pruning
            metric_reshaped = ops.reshape(metric, (metric.shape[0], -1))
            weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.sparsity)
            self.mask = ops.reshape(mask, weight.shape)

    def collect_input(self, x, weight, training):
        if self.done or not training:
            return
        """
            Accumulates layer inputs starting at step t_start_collecting for t_delta steps, then averages it.
            Calculates a metric based on weight absolute values and norm of inputs.
            For linear layers, calculate norm over batch dimension.
            For conv layers, take average over batch dimension and calculate norm over flattened kernel_size dimension.
            If N and M are defined, do N:M pruning.
            """
        ok_batch = True
        if self.inputs is not None:
            batch_size = self.inputs.shape[0]
            ok_batch = x.shape[0] == batch_size
        if not training or not ok_batch:
            # Don't collect during validation
            return
        if self.t < self.t_start_collecting_batch:
            return
        self.batches_collected += 1
        self.total += 1

        self.inputs = x if self.inputs is None else self.inputs + x
        if self.batches_collected % (self.config["pruning_parameters"]["t_delta"]) == 0:
            inputs_avg = self.inputs / self.total
            self.prune(inputs_avg, weight)
            self.done = True
            self.inputs = None

    def prune(self, x, weight):
        if self.layer_type == "linear":
            self.handle_linear(x, weight)
        else:
            self.handle_conv(x, weight)

    def call(self, weight):  # Mask is only updated every t_delta step, using collect_output
        return self.mask * weight

    def post_pre_train_function(self):
        self.is_pretraining = False

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def post_round_function(self):
        pass

    def pre_finetune_function(self):
        pass

    def calculate_additional_loss(self):
        return 0

    def get_layer_sparsity(self, weight):
        pass

    def get_hard_mask(self, weight=None):
        return self.mask

    def post_epoch_function(self, epoch, total_epochs):
        if self.is_pretraining is False:
            self.t += 1
        pass
