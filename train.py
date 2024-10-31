import os
os.environ["KERAS_BACKEND"] = "torch"
import torch.nn as nn
import torch
import torchvision
import tqdm

from sparse_layers import  get_layer_keep_ratio, get_model_losses, add_pruning_to_model, \
post_epoch_functions, post_round_functions, save_weights_functions, rewind_weights_functions, \
pre_finetune_functions, post_pretrain_functions, pre_epoch_functions, remove_pruning_from_model
from utils import get_scheduler, get_optimizer
from parser import get_parser
from data import get_cifar10_data
from torch.utils.tensorboard import SummaryWriter
import keras_core as keras
keras.backend.set_image_data_format('channels_first')



def get_model(config, device):
    if config.model == "resnet18":
        return torchvision.models.resnet18().to(device)
    elif config.model == "resnet34":
        return torchvision.models.resnet34().to(device)
    elif config.model == "resnet50":
        return torchvision.models.resnet50().to(device)
    elif config.model == "vgg16":
        return torchvision.models.vgg16().to(device)

def call_post_round_functions(model, config, round):
        if config.rewind == "round":
            rewind_weights_functions(model)
        elif config.rewind == "post-ticket-search" and round == config.rounds - 1:
            rewind_weights_functions(model)
        if round < config.rounds - 1:
            post_round_functions(model)


def plot_training_loss(loss, losses, config, global_step):
    if global_step % config.plot_frequency == 0:
        writer.add_scalar(f"train_output_loss", loss.item(), global_step)
        writer.add_scalar(f"train_sparse_loss", losses, global_step)


def validation(model, testloader, device, criterion, global_step):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        ratio = get_layer_keep_ratio(model)
        writer.add_scalar(f"validation_output_loss", loss.item(), global_step)
        writer.add_scalar(f"validation_sparse_loss", losses, global_step)
        writer.add_scalar(f"validation_acc", correct / total, global_step)
        writer.add_scalar(f"validation_remaining_weights", ratio, global_step)


def iterative_train(model, config, trainloader, testloader, device, criterion, writer):
    global_step = torch.tensor(0)
    if config.pretraining_epochs:
        epochs = config.epochs
        config.epochs = config.pretraining_epochs
        train(model, config, trainloader, testloader, device, criterion, -1, writer, global_step)
        config.epochs = epochs
        post_pretrain_functions(model, config)
        print('Pretraining finished')

    for r in range(config.rounds):
        train(model, config, trainloader, testloader, device, criterion, r, writer, global_step)
        call_post_round_functions(model, config, r)
    print("Training finished")
    if config.fine_tune:
        global_step = torch.tensor(0)
        pre_finetune_functions(model)
        train(model, config, trainloader, testloader, device, criterion, config.rounds, writer, global_step)
        print("Fine-tuning finished")
    return model


def train(model, config, trainloader, testloader, device, criterion, round, writer, global_step):
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(optimizer, config)
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        pre_epoch_functions(model, epoch, config.epochs)
        if round == 0 and config.save_weights_epoch == epoch:
            save_weights_functions(model)
        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            plot_training_loss(loss, losses, config, global_step)
            loss += losses
            loss.backward()
            optimizer.step()
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        validation(model, testloader, device, criterion, global_step)
        post_epoch_functions(model, epoch, config.epochs)
    return model


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    comment = f"_{config.model}_{config.pruning_method}"
    writer = SummaryWriter(comment=comment)
    output_dir = writer.get_logdir()
    model = get_model(config, device)
    
    sparse_model = add_pruning_to_model(model, config)
    sparse_model = sparse_model.to(device)
    from torchsummary import summary
    summary(sparse_model, (3,32,32), device=device)
    trainloader, testloader = get_cifar10_data(config.batch_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    trained_sparse_model = iterative_train(sparse_model, config, trainloader, testloader, device, criterion, writer)
    trained_model = remove_pruning_from_model(sparse_model, config)
    summary(trained_model, (3,32,32))
    #torch.save(trained_model.state_dict(), "test_model.pt")
    torch.save(trained_model, f"{output_dir}/model.pth")
    

