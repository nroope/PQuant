import keras


@keras.saving.register_keras_serializable(package="Layers")
class FITCompress(keras.layers.Layer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.finetuning import TuningConfig

            config = TuningConfig.load_from_config(config)
        self.config = config
        self.is_pretraining = True
        self.is_finetuning = False

    def build(self, input_shape):
        self.mask = self.add_weight(shape=input_shape, initializer="ones", trainable=False)
        super().build(input_shape)

    def call(self, weight):
        return self.mask * weight

    def get_hard_mask(self, weight=None):
        return self.mask

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def calculate_additional_loss(*args, **kwargs):
        return 0

    def pre_finetune_function(self):
        self.is_finetuning = True

    def post_round_function(self):
        pass

    def post_pre_train_function(self):
        self.is_pretraining = False

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "config": self.config.get_dict(),
            }
        )
        return config
