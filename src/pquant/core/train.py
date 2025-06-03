import keras


def iterative_train(model, config, train_func, valid_func, **kwargs):
    if keras.backend.backend() == "torch":
        from pquant.core.torch_impl.train_torch import iterative_train_torch

        return iterative_train_torch(model, config, train_func, valid_func, **kwargs)
    else:
        from pquant.core.tf_impl.train_tf import iterative_train_tf

        return iterative_train_tf(model, config, train_func, valid_func, **kwargs)
