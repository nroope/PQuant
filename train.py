import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import torch.nn as nn
from argparse import Namespace
from sparse_layers import  add_pruning_to_model, \
post_epoch_functions, post_round_functions, save_weights_functions, rewind_weights_functions, \
pre_finetune_functions, post_pretrain_functions, pre_epoch_functions, remove_pruning_from_model, get_model_losses, get_layer_keep_ratio
from utils import get_scheduler, get_optimizer, plot_weights_per_layer
from parser import parse_cmdline_args, write_config_to_yaml
from weaver.utils.nn.tools import TensorboardHelper
import keras_core as keras
from weaver.utils.nn.tools import train_classification, evaluate_classification
import tqdm
from smartpixels import custom_loss
import numpy as np
from torchinfo import summary
from resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16
from smartpixels import get_smartpixel_data_and_model

from data import get_cifar10_data, get_imagenet_data
from weaver.train import train_load, model_setup
keras.backend.set_image_data_format('channels_first')


def call_post_round_functions(model, rewind, rounds, r):
        if rewind == "round":
            rewind_weights_functions(model)
        elif rewind == "post-ticket-search" and r == rounds - 1:
            rewind_weights_functions(model)
        else:
            post_round_functions(model)

########################################################
############ Load models, datasets etc. ################
########################################################

def get_model_data_loss_func(config, device):
    if "resnet" in config.model:
        model, train_loader, val_loader, loss_func = get_resnet_model_data(config, device)   
    elif config.model == "particle_transformer":
        train_loader, val_loader, data_config, _, _ = train_load(config)
        model, _, loss_func = model_setup(config, data_config, device=device)
    elif config.model == "smartpixel":
        model, train_loader, val_loader, loss_func = get_smartpixel_data_and_model()
    model = model.to(device)
    return model, train_loader, val_loader, loss_func

def get_resnet_model_data(config, device):
    if config.dataset == "imagenet":
        if config.model == "resnet18":
            model = resnet18().to(device)
        elif config.model == "resnet34":
            model = resnet34().to(device)
        elif config.model == "resnet50":
            model = resnet50().to(device)
        elif config.model == "resnet101":
            model = resnet101().to(device)
        elif config.model == "resnet152":
            model = resnet152().to(device)
        elif config.model == "vgg16":
            model = vgg16().to(device)
        summary(model, (1,3,224,224))
    elif config.dataset == "cifar10":
        if config.model == "resnet20":
            model = resnet20().to(device)
        elif config.model == "resnet32":
            model = resnet32().to(device)
        elif config.model == "resnet44":
            model = resnet44().to(device)
        elif config.model == "resnet56":
            model = resnet56().to(device)
        elif config.model == "resnet110":
            model = resnet110().to(device)
        elif config.model == "resnet1202":
            model = resnet1202().to(device)
        summary(model, (1,3,32,32))
    if config.dataset == "cifar10":
        train_loader, val_loader = get_cifar10_data(config.batch_size)
    elif config.dataset == "imagenet":
        train_loader, val_loader = get_imagenet_data(config)
    loss_func = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    return model, train_loader, val_loader, loss_func


########################################################
################ Training loops ########################
########################################################
def train_smartpixel(model, train_data, optimizer, device, epoch, writer=None, *args, **kwargs):
    for (inputs, target) in tqdm.tqdm(train_data):
        inputs = torch.tensor(inputs.numpy())
        target = torch.tensor(target.numpy())
            
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = custom_loss(outputs, target)
        losses = get_model_losses(model, torch.tensor(0.).to(device))
        loss += losses
        loss.backward()
        optimizer.step()
    if writer is not None:
        writer.write_scalars([("training_loss", loss.item(),  epoch)])

def validate_smartpixel(model, validation_data, device, epoch, writer=None, *args, **kwargs):
    for (inputs, target) in (validation_data):
        inputs = torch.tensor(inputs.numpy())
        target = torch.tensor(target.numpy())
            
        inputs = inputs.to(device)
        target = target.to(device)
        outputs = model(inputs)
        loss = custom_loss(outputs, target)
        losses = get_model_losses(model, torch.tensor(0.).to(device))
        loss += losses
    ratio = get_layer_keep_ratio(model)
    if writer is not None:
        writer.write_scalars([("validation_remaining_weights", ratio, epoch)])
        writer.write_scalars([("validation_loss", loss.item(), epoch)])
    

def autosparse_autotune_resnet(model, sparse_model, config, trainloader, testloader, device, loss_func, writer):
    # WIP AutoSparse alpha-autotuning training, for ResNets
    if type(config) is dict:
        config = Namespace(**config)
    optimizer = get_optimizer(config, sparse_model)
    scheduler = get_scheduler(optimizer, config)
    global_step = torch.tensor(0)
    dense_losses = []
    autotune_epochs = config.autotune_epochs
    print("TRAINING DENSE")
    for epoch in range(autotune_epochs): 
        model.train()
        dense_losses_avg = []
        pre_epoch_functions(model, epoch, config.epochs)
        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            dense_losses_avg.append(loss.item())
            
            loss.backward()
            optimizer.step()
            global_step += 1
        dense_losses.append(np.mean(dense_losses_avg))
    optimizer = get_optimizer(config, sparse_model)
    scheduler = get_scheduler(optimizer, config)
    print("TRAINING SPARSE")
    for epoch in range(config.epochs):
        sparse_model.train()
        outputs_avg = []
        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = sparse_model(inputs)
            loss = loss_func(outputs, labels)
            losses = get_model_losses(sparse_model, torch.tensor(0.).to(device))
            plot_training_loss(loss, losses, global_step, writer)
            loss += losses
            outputs_avg.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1

        if scheduler is not None:
            scheduler.step()
        validate_resnet(sparse_model, testloader, device, loss_func, global_step, writer)
        loss_avg = np.mean(outputs_avg)
        alpha_multiplier = 1.
        if epoch < autotune_epochs: 
                dense_loss = float(dense_losses[epoch]) 
                eps_0 = 0.01 
                eps_1 = 0.05 
                eps_2 = 0.005 
                if loss_avg > dense_loss * (1.0 + eps_0): 
                    alpha_multiplier = 1.0 + eps_1
                else: 
                    alpha_multiplier = 1.0 - eps_2
        post_epoch_functions(sparse_model, epoch, config.epochs, alpha_multiplier=alpha_multiplier, autotune_epochs=autotune_epochs, writer=writer, global_step=global_step)
    return sparse_model

def plot_training_loss(loss, losses, global_step, writer):
        writer.write_scalars([(f"train_output_loss", loss.item(), global_step)])
        writer.write_scalars([(f"train_sparse_loss", losses, global_step)])

def train_particle_transformer(model, steps_per_epoch, train_loader, loss_func, dev, writer, epoch, optimizer, scheduler, *args, **kwargs):
    """Train particle transformer for 1 epoch"""
    train_classification(model, loss_func, optimizer, scheduler, train_loader, dev, epoch, steps_per_epoch, None, writer)

def validate_particle_transformer(model, steps_per_epoch_val, test_loader, loss_func, dev, writer, epoch, *args, **kwargs):
    evaluate_classification(model, test_loader, dev, epoch, True, loss_func, steps_per_epoch_val, tb_helper=writer)

def train_resnet(model, trainloader, device, loss_func, writer, epoch, optimizer, scheduler, *args, **kwargs):
    """ Train ResNets for 1 epoch """
    for data in tqdm.tqdm(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        losses = get_model_losses(model, torch.tensor(0.).to(device))
        loss += losses
        loss.backward()
        optimizer.step()
        epoch += 1
        if scheduler is not None:
            scheduler.step()
    plot_training_loss(loss, losses, epoch, writer)

def validate_resnet(model, testloader, device, loss_func, epoch, writer, *args, **kwargs):
    """Validation loop for ResNets"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if loss_func is not None:
                loss = loss_func(outputs, labels)
                losses = get_model_losses(model, torch.tensor(0.).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        ratio = get_layer_keep_ratio(model)
        if writer is not None:
            writer.write_scalars([(f"validation_output_loss", loss.item(), epoch)])
            writer.write_scalars([(f"validation_sparse_loss", losses, epoch)])
            writer.write_scalars([(f"validation_acc", correct / total, epoch)])
            writer.write_scalars([(f"validation_remaining_weights", ratio, epoch)])
        
########################################################
########################################################
########################################################

def iterative_train(model, config, train_func, valid_func, *args, **kwargs):
    """ 
    Generic training loop, user provides training and validation functions
    """
    epoch = torch.tensor(0) # Keeps track of all the epochs completed
    if config.pretraining_epochs > 0:
        for e in range(config.pretraining_epochs):
            model.train()
            pre_epoch_functions(model, e, config.pretraining_epochs)
            train_func(model, epoch=epoch, *args, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, *args, **kwargs)
            post_epoch_functions(model, e, config.pretraining_epochs)
            epoch += 1
        post_pretrain_functions(model, config)
        print('Pretraining finished')
    for r in range(config.rounds):
        for e in range(config.epochs):
            model.train()
            if r == 0 and config.save_weights_epoch == e:
                save_weights_functions(model)
            pre_epoch_functions(model, e, config.epochs)
            train_func(model, epoch=epoch, *args, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, *args, **kwargs)
            post_epoch_functions(model, e, config.epochs)
            epoch += 1
        call_post_round_functions(model, config.rewind, config.rounds, r)
    print("Training finished")
    if config.fine_tuning_epochs > 0:
        pre_finetune_functions(model)
        for e in range(config.fine_tuning_epochs):
            model.train()
            pre_epoch_functions(model, e, config.fine_tuning_epochs)
            train_func(model, epoch=epoch, *args, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, *args, **kwargs)
            post_epoch_functions(model, e, config.fine_tuning_epochs)
            epoch += 1
        print("Fine-tuning finished")
    return model


def main(config):
    if type(config) is dict:
        config = Namespace(**config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    comment = f"_{config.model}_{config.pruning_method}"
    writer = TensorboardHelper(tb_comment=comment, tb_custom_fn=None)
    output_dir = writer.writer.get_logdir()
    write_config_to_yaml(config, output_dir)
    sparse_model, train_loader, val_loader, loss_func = get_model_data_loss_func(config, device)
    if config.do_pruning:
        sparse_model = add_pruning_to_model(sparse_model, config)
    if config.pruning_method == "autosparse" and "resnet" in config.model: # WIP, use only for resnets
        model, _, _, _ = get_model_data_loss_func(config, device)
        trained_sparse_model = autosparse_autotune_resnet(model, sparse_model, config, train_loader, val_loader, device, loss_func, writer)
    elif config.model == "smartpixel":
        optimizer = get_optimizer(config, sparse_model)
        trained_sparse_model = iterative_train(model = sparse_model, config = config, train_func = train_smartpixel, 
                                                valid_func = validate_smartpixel, train_data = train_loader, validation_data = val_loader, 
                                                device = device, writer = writer, 
                                                optimizer = optimizer
                                                )
    elif "resnet" in config.model:
        optimizer = get_optimizer(config, sparse_model)
        scheduler = get_scheduler(optimizer, config)
        trained_sparse_model = iterative_train(model = sparse_model, config = config, train_func = train_resnet, 
                                                valid_func = validate_resnet, trainloader = train_loader, testloader = val_loader, 
                                                device = device, loss_func = loss_func, writer = writer, 
                                                optimizer = optimizer, scheduler = scheduler
                                                )
    elif config.model == "particle_transformer":
        optimizer = get_optimizer(config, sparse_model)
        scheduler = get_scheduler(optimizer, config)
        trained_sparse_model = iterative_train(model = sparse_model, config=config, train_loader=train_loader, 
                                               test_loader=val_loader, dev=device, train_func=train_particle_transformer, valid_func=validate_particle_transformer, 
                                               loss_func=loss_func, writer=writer, steps_per_epoch = config.steps_per_epoch,
                                               steps_per_epoch_val = config.steps_per_epoch_val, optimizer = optimizer,
                                               scheduler = scheduler)

    torch.save(trained_sparse_model.state_dict(), f"{output_dir}/final_model_before_remove_pruning_layers.pt")
    sparse_model = remove_pruning_from_model(trained_sparse_model, config)
    torch.save(sparse_model.state_dict(), f"{output_dir}/final_model.pt")
    plot_weights_per_layer(sparse_model, output_dir)

if __name__ == "__main__":
    config = parse_cmdline_args()
    main(config)

