import keras
from keras import ops
from keras.initializers import Constant


@keras.saving.register_keras_serializable(package="PQuant")
class ContinuousSparsification(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.config = config
        self.final_temp = config.pruning_parameters.final_temp
        self.do_hard_mask = False
        self.layer_type = layer_type
        self.is_pretraining = True

    def build(self, input_shape):
        self.s_init = ops.convert_to_tensor(self.config.pruning_parameters.threshold_init * ops.ones(input_shape))
        self.s = self.add_weight(name="threshold", shape=input_shape, initializer=Constant(self.s_init), trainable=True)
        self.scaling = 1.0 / ops.sigmoid(self.s_init)
        self.beta = self.add_weight(name="beta", shape=(), initializer=Constant(1.0), trainable=False)
        self.mask = self.add_weight(name="mask", shape=input_shape, initializer=Constant(1.0), trainable=False)
        super().build(input_shape)

    def call(self, weight):
        if self.is_pretraining:
            return weight
        mask = self.get_mask()
        self.mask.assign(mask)
        return mask * weight

    def pre_finetune_function(self):
        self.do_hard_mask = True

    def get_mask(self):
        if self.do_hard_mask:
            mask = self.get_hard_mask()
            return mask
        else:
            mask = ops.sigmoid(self.beta * self.s)
            mask = mask * self.scaling
            return mask

    def post_pre_train_function(self):
        self.is_pretraining = False

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def post_epoch_function(self, epoch, total_epochs):
        self.beta.assign(self.beta * self.final_temp ** (1 / (total_epochs - 1)))

    def get_hard_mask(self, weight=None):
        if self.config.pruning_parameters.enable_pruning:
            return ops.cast((self.s > 0), self.s.dtype)
        return ops.convert_to_tensor(1.0)

    def post_round_function(self):
        min_beta_s_s0 = ops.minimum(self.beta * self.s, self.s_init)
        self.s.assign(min_beta_s_s0)
        self.beta.assign(1.0)

    def calculate_additional_loss(self):
        return ops.convert_to_tensor(
            self.config.pruning_parameters.threshold_decay * ops.norm(ops.ravel(self.get_mask()), ord=1)
        )

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask()) / ops.size(weight)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "config": self.config.get_dict(),
                "layer_type": self.layer_type,
            }
        )
        return config
