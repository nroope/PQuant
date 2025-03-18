import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import torch.nn as nn

from pquant.core.compressed_layers import  add_pruning_and_quantization, remove_pruning_from_model
from pquant.core.utils import write_config_to_yaml
from pquant.core.train import iterative_train
from optimizers import get_scheduler, get_optimizer, plot_weights_per_layer
from parser import parse_cmdline_args
from torch.utils.tensorboard import SummaryWriter
import keras_core as keras
from resnet import train_resnet, validate_resnet, get_resnet_model
from smartpixels import get_smartpixel_data_and_model, train_smartpixel, validate_smartpixel
from data import get_cifar10_data, get_imagenet_data
keras.backend.set_image_data_format('channels_first')


def get_model_data_loss_func(config, device):
    if "resnet" in config["model"] or "vgg" in config["model"]:
        model, train_loader, val_loader, loss_func = get_resnet_model_data(config, device)   
    elif config["model"] == "smartpixel":
        model, train_loader, val_loader, loss_func = get_smartpixel_data_and_model()
    model = model.to(device)
    return model, train_loader, val_loader, loss_func

def get_resnet_model_data(config, device):
    model = get_resnet_model(config, device)
    if config["dataset"] == "cifar10":
        train_loader, val_loader = get_cifar10_data(config["batch_size"])
    elif config["dataset"] == "imagenet":
        train_loader, val_loader = get_imagenet_data(config)
    loss_func = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    return model, train_loader, val_loader, loss_func

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    comment = f"_{config["model"]}_{config["pruning_parameters"]["pruning_method"]}"
    writer = SummaryWriter(comment=comment)
    output_dir = writer.get_logdir()
    write_config_to_yaml(config, f"{output_dir}/config.yaml")
    sparse_model, train_loader, val_loader, loss_func = get_model_data_loss_func(config, device)
    sparse_model = add_pruning_and_quantization(sparse_model, config)
    sparse_model = sparse_model.to(device)
    if config["model"] == "smartpixel":
        optimizer = get_optimizer(config, sparse_model)
        trained_sparse_model = iterative_train(model = sparse_model, config = config, train_func = train_smartpixel, 
                                                valid_func = validate_smartpixel, train_data = train_loader, validation_data = val_loader, 
                                                device = device, writer = writer, 
                                                optimizer = optimizer
                                                )
    elif "resnet" in config["model"] or "vgg" in config["model"]:
        optimizer = get_optimizer(config, sparse_model)
        scheduler = get_scheduler(optimizer, config)
        trained_sparse_model = iterative_train(model = sparse_model, config = config, train_func = train_resnet, 
                                                valid_func = validate_resnet, trainloader = train_loader, testloader = val_loader, 
                                                device = device, loss_func = loss_func, writer = writer, 
                                                optimizer = optimizer, scheduler = scheduler
                                                )

    torch.save(trained_sparse_model.state_dict(), f"{output_dir}/final_model_before_remove_pruning_layers.pt")
    sparse_model = remove_pruning_from_model(trained_sparse_model, config)
    torch.save(sparse_model.state_dict(), f"{output_dir}/final_model.pt")
    plot_weights_per_layer(sparse_model, output_dir)

if __name__ == "__main__":
    config = parse_cmdline_args()
    main(config)


