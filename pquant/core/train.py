import torch
from pquant.core.compressed_layers import post_epoch_functions, save_weights_functions, pre_finetune_functions, \
post_pretrain_functions, call_post_round_functions, pre_epoch_functions, get_layer_keep_ratio

        
def iterative_train(model, config, train_func, valid_func, *args, **kwargs):
    """ 
    Generic training loop, user provides training and validation functions
    """
    epoch = torch.tensor(0) # Keeps track of all the epochs completed
    training_config = config["training_parameters"]
    if training_config["pretraining_epochs"] > 0:
        for e in range(training_config["pretraining_epochs"]):
            model.train()
            pre_epoch_functions(model, e, training_config["pretraining_epochs"])
            train_func(model, epoch=epoch, *args, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, *args, **kwargs)
            post_epoch_functions(model, e, training_config["pretraining_epochs"])
            epoch += 1
        post_pretrain_functions(model, config)
        print('Pretraining finished')
    for r in range(training_config["rounds"]):
        for e in range(training_config["epochs"]):
            model.train()
            if r == 0 and training_config["save_weights_epoch"] == e:
                save_weights_functions(model)
            pre_epoch_functions(model, e, training_config["epochs"])
            train_func(model, epoch=epoch, *args, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, *args, **kwargs)
            post_epoch_functions(model, e, training_config["epochs"])
            epoch += 1
        call_post_round_functions(model, training_config["rewind"], training_config["rounds"], r)
    print("Training finished")
    pre_finetune_functions(model)
    if training_config["fine_tuning_epochs"] > 0:
        for e in range(training_config["fine_tuning_epochs"]):
            model.train()
            pre_epoch_functions(model, e, training_config["fine_tuning_epochs"])
            train_func(model, epoch=epoch, *args, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, *args, **kwargs)
            post_epoch_functions(model, e, training_config["fine_tuning_epochs"])
            epoch += 1
        print("Fine-tuning finished")
    # Final score
    return model

