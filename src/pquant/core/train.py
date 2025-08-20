from pquant.core.utils import get_backend


def iterative_train(model, config, train_func, valid_func, **kwargs):
    backend = get_backend()
    return backend.iterative_train(model, config, train_func, valid_func, **kwargs)
