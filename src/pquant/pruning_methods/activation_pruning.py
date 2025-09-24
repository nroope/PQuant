import keras
from keras import ops


class ActivationPruning(keras.layers.Layer):

    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.act_type = "relu"
        self.t = 0
        self.batches_collected = 0
        self.layer_type = layer_type
        self.activations = None
        self.total = 0.0
        self.is_pretraining = True
        self.done = False
        self.threshold = ops.convert_to_tensor(config["pruning_parameters"]["threshold"])
        self.t_start_collecting_batch = self.config["pruning_parameters"]["t_start_collecting_batch"]

    def build(self, input_shape):
        self.shape = (input_shape[0], 1)
        if self.layer_type == "conv":
            if len(input_shape) == 3:
                self.shape = (input_shape[0], 1, 1)
            else:
                self.shape = (input_shape[0], 1, 1, 1)
        self.mask = ops.ones(self.shape)

    def collect_output(self, output, training):
        """
        Accumulates values for how often the outputs of the neurons and channels of
        linear/convolution layer are over 0. Every t_delta steps, uses these values to update
        the mask to prune those channels and neurons that are active less than a given threshold
        """
        if self.done or not training or self.is_pretraining:
            # Don't collect during validation
            return
        if self.activations is None:
            # Initialize activations dynamically
            self.activations = ops.zeros(shape=output.shape[1:], dtype=output.dtype)
        if self.t < self.t_start_collecting_batch:
            return
        self.batches_collected += 1
        self.total += output.shape[0]
        gt_zero = ops.cast((output > 0), output.dtype)
        gt_zero = ops.sum(gt_zero, axis=0)  # Sum over batch, take average during mask update
        self.activations += gt_zero
        if self.batches_collected % self.config["pruning_parameters"]["t_delta"] == 0:
            pct_active = self.activations / self.total
            self.t = 0
            self.total = 0
            if self.layer_type == "linear":
                self.mask = ops.expand_dims(ops.cast((pct_active > self.threshold), pct_active.dtype), 1)
            else:
                pct_active = ops.reshape(pct_active, (pct_active.shape[0], -1))
                pct_active_avg = ops.mean(pct_active, axis=-1)
                pct_active_above_threshold = ops.cast((pct_active_avg > self.threshold), pct_active_avg.dtype)
                if len(output.shape) == 3:
                    self.mask = ops.reshape(pct_active_above_threshold, list(pct_active_above_threshold.shape) + [1, 1])
                else:
                    self.mask = ops.reshape(pct_active_above_threshold, list(pct_active_above_threshold.shape) + [1, 1, 1])
            self.activations *= 0.0
            self.done = True

    def call(self, weight):  # Mask is only updated every t_delta step, using collect_output
        if self.is_pretraining and self.config["fitcompress_parameters"]["enable_fitcompress"]:
            return weight
        else:
            return self.mask * weight

    def get_hard_mask(self, weight):
        return self.mask

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

    def post_epoch_function(self, epoch, total_epochs):
        if self.is_pretraining is False:
            self.t += 1
        pass
