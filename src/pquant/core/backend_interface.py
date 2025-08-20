from abc import ABC, abstractmethod


class BackendInterface(ABC):
    @abstractmethod
    def add_default_layer_quantization_pruning_to_config(self, model, config):
        pass

    @abstractmethod
    def iterative_train(self, model, config, train_func, valid_func, **kwargs):
        pass

    @abstractmethod
    def remove_pruning_from_model(self, model, config):
        pass

    @abstractmethod
    def add_compression_layers(self, model, config, input_shape=None):
        pass

    @abstractmethod
    def post_epoch_functions(self, model, epoch, total_epochs, **kwargs):
        pass

    @abstractmethod
    def post_pretrain_functions(self, model, config):
        pass

    @abstractmethod
    def pre_epoch_functions(self, model, epoch, total_epochs):
        pass

    @abstractmethod
    def pre_finetune_functions(self, model):
        pass

    @abstractmethod
    def save_weights_functions(self, model):
        pass

    @abstractmethod
    def get_layer_keep_ratio(self, model):
        pass

    @abstractmethod
    def get_model_losses(self, model, losses):
        pass

    def call_post_round_functions(self, model, rewind, rounds, r):
        if rewind == "round":
            self.rewind_weights_functions(model)
        elif rewind == "post-ticket-search" and r == rounds - 1:
            self.rewind_weights_functions(model)
        else:
            self.post_round_functions(model)
