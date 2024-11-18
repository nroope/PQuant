import os
os.environ["KERAS_BACKEND"] = "torch"
import torch.nn as nn
import torch
import tqdm
import numpy as np
from argparse import Namespace
from sparse_layers import  get_layer_keep_ratio, get_model_losses, add_pruning_to_model, \
post_epoch_functions, post_round_functions, save_weights_functions, rewind_weights_functions, \
pre_finetune_functions, post_pretrain_functions, pre_epoch_functions, remove_pruning_from_model
from utils import get_scheduler, get_optimizer
from parser import parse_cmdline_args, write_config_to_yaml
from data import get_cifar10_data, get_imagenet_data
from weaver.train import train_load, model_setup
from weaver.utils.nn.tools import train_classification, evaluate_classification, TensorboardHelper
import keras_core as keras
keras.backend.set_image_data_format('channels_first')
from torchinfo import summary
from resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16

def get_resnet_model(config, device):
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

    return model

def call_post_round_functions(model, config, round):
        if config.rewind == "round":
            rewind_weights_functions(model)
        elif config.rewind == "post-ticket-search" and round == config.rounds - 1:
            rewind_weights_functions(model)
        if round < config.rounds - 1:
            post_round_functions(model)


def plot_training_loss(loss, losses, config, global_step, writer):
    if global_step % config.plot_frequency == 0:
        writer.write_scalars([(f"train_output_loss", loss.item(), global_step)])
        writer.write_scalars([(f"train_sparse_loss", losses, global_step)])


def validation(model, testloader, device, criterion, global_step, writer):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                losses = get_model_losses(model, torch.tensor(0.).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        ratio = get_layer_keep_ratio(model)
        if writer is not None:
            writer.write_scalars([(f"validation_output_loss", loss.item(), global_step)])
            writer.write_scalars([(f"validation_sparse_loss", losses, global_step)])
            writer.write_scalars([(f"validation_acc", correct / total, global_step)])
            writer.write_scalars([(f"validation_remaining_weights", ratio, global_step)])


# Built on top of weaver training loop
def iterative_train_parT(model, config, output_dir, trainloader, testloader, device, loss_func, writer):
    if type(config) is dict:
        config = Namespace(**config)
    global_step = torch.tensor(0)
    if config.pretraining_epochs:
        epochs = config.epochs
        config.epochs = config.pretraining_epochs
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(optimizer, config)
        for epoch in range(config.epochs):  
            pre_epoch_functions(model, epoch, config.epochs)
            train_classification(model, loss_func, optimizer, scheduler, trainloader, device, 0, config.steps_per_epoch, None, writer)
            global_step += 1
        config.epochs = epochs
        post_pretrain_functions(model, config)
        print('Pretraining finished')
    for r in range(config.rounds):
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(optimizer, config)
        for epoch in range(config.epochs):
            if r == 0 and config.save_weights_epoch == epoch:
                save_weights_functions(model)   
            pre_epoch_functions(model, epoch, config.epochs) 
            train_classification(model, loss_func, optimizer, scheduler , trainloader, device, global_step, config.steps_per_epoch, None, writer)
            evaluate_classification(model, testloader, device, global_step, True, loss_func, config.steps_per_epoch_val, tb_helper=writer)
            post_epoch_functions(model, epoch, config.epochs)
            ratio = get_layer_keep_ratio(model)
            writer.write_scalars([("validation_remaining_weights", ratio,  global_step)])
            global_step += 1
        torch.save(model.state_dict(), f"{output_dir}/pre_post_round_{r}.pt")
        call_post_round_functions(model, config, r)
    print("Training finished")
    if config.fine_tune:
        pre_finetune_functions(model)
        scheduler = get_scheduler(optimizer, config)
        for epoch in range(config.epochs):    
            pre_epoch_functions(model, epoch, config.epochs)
            train_classification(model, loss_func, optimizer, scheduler, trainloader, device, global_step, config.steps_per_epoch, None, writer)
            evaluate_classification(model, testloader, device, global_step, True, loss_func, config.steps_per_epoch_val, tb_helper=writer)
            global_step += 1
    print("Fine-tuning finished")
    return model


def iterative_train(model, config, trainloader, testloader, device, loss_func, writer):
    if type(config) is dict:
        config = Namespace(**config)
    global_step = torch.tensor(0)
    if config.pretraining_epochs:
        epochs = config.epochs
        config.epochs = config.pretraining_epochs
        train(model, config, trainloader, testloader, device, loss_func, -1, writer, global_step)
        config.epochs = epochs
        post_pretrain_functions(model, config)
        print('Pretraining finished')
    for r in range(config.rounds):
        train(model, config, trainloader, testloader, device, loss_func, r, writer, global_step)
        call_post_round_functions(model, config, r)
        torch.save(model.state_dict(), f"{writer.writer.get_logdir()}/model_round_{r}.pt")
    print("Training finished")
    if config.fine_tune:
        pre_finetune_functions(model)
        train(model, config, trainloader, testloader, device, loss_func, config.rounds, writer, global_step)
        print("Fine-tuning finished")
    return model

def train(model, config, trainloader, testloader, device, loss_func, r, writer, global_step):
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)
    for epoch in range(config.epochs):
        model.train()
        pre_epoch_functions(model, epoch, config.epochs)
        if r == 0 and config.save_weights_epoch == epoch:
            save_weights_functions(model)
        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            plot_training_loss(loss, losses, config, global_step, writer)
            loss += losses
            loss.backward()
            optimizer.step()
            global_step += 1

        if scheduler is not None:
            scheduler.step()
        validation(model, testloader, device, loss_func, global_step, writer)
        post_epoch_functions(model, epoch, config.epochs)
    return model


def autosparse_autotune(model, sparse_model, config, trainloader, testloader, device, loss_func, writer):
    # WIP AutoSparse alpha-autotuning training
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)
    if type(config) is dict:
        config = Namespace(**config)
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
            plot_training_loss(loss, 0., config, global_step, writer)
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
            plot_training_loss(loss, losses, config, global_step, writer)
            loss += losses
            outputs_avg.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1

        if scheduler is not None:
            scheduler.step()
        validation(sparse_model, testloader, device, loss_func, global_step, writer)
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


def get_model_data_loss_func(config, device):
    if config.dataset == "cifar10":
        train_loader, val_loader = get_cifar10_data(config.batch_size)
    elif config.dataset == "imagenet":
        train_loader, val_loader = get_imagenet_data(config)
    else:
        train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(config)
    if config.model == "parT":
        model, model_info, loss_func = model_setup(config, data_config, device=device)
    else:
        model = get_resnet_model(config, device)   
        loss_func = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    return model, train_loader, val_loader, loss_func

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
    sparse_model = sparse_model.to(device)
    if config.model == "parT":
        trained_sparse_model = iterative_train_parT(sparse_model, config, output_dir, train_loader, val_loader, device, loss_func, writer)
    elif config.pruning_method == "autosparse": # WIP, use only for resnets
        model = get_resnet_model(config, device)
        trained_sparse_model = autosparse_autotune(model, sparse_model, config, train_loader, val_loader, device, loss_func, writer)
    else:
        trained_sparse_model = iterative_train(sparse_model, config, train_loader, val_loader, device, loss_func, writer)
    sparse_model = remove_pruning_from_model(trained_sparse_model, config)
    torch.save(sparse_model.state_dict(), f"{output_dir}/final_model.pt")
    

if __name__ == "__main__":
    config = parse_cmdline_args()
    main(config)


