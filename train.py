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
import keras_core as keras
keras.backend.set_image_data_format('channels_first')


def call_post_round_functions(model, config, round):
        if config.rewind == "round":
            rewind_weights_functions(model)
        elif config.rewind == "post-ticket-search" and round == config.rounds - 1:
            rewind_weights_functions(model)
        if round < config.rounds - 1:
            post_round_functions(model)


def test(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        get_layer_keep_ratio(model)

def iterative_train(model, config, trainloader, testloader, device, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if config.pretraining_epochs:
        epochs = config.epochs
        config.epochs = config.pretraining_epochs
        train(model, config, trainloader, testloader, device, criterion, -1)
        config.epochs = epochs
        post_pretrain_functions(model, config)
    for r in range(config.rounds):
        train(model, config, trainloader, testloader, device, criterion, r)
        call_post_round_functions(model, config, r)
    if config.fine_tune:
        pre_finetune_functions(model)
        train(model, config, trainloader, testloader, device, criterion, config.rounds)
    return model

def train(model, config, trainloader, testloader, device, criterion=None, round=0):
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
            loss += losses
            if config.pruning_method == "cs":
                # L1-regularization used in CS
                for n, v in model.named_parameters():
                    if "pruning_layer" not in n:
                        loss += 0.0001 * v.abs().sum()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        test(model, testloader, device)
        post_epoch_functions(model, epoch, config.epochs)
    print('Finished Training')
    return model


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet18().to(device)
    sparse_model = add_pruning_to_model(model, config)
    sparse_model = sparse_model.to(device)
    from torchsummary import summary
    summary(sparse_model, (3,32,32), device=device)
    trainloader, testloader = get_cifar10_data(config.batch_size)
    trained_sparse_model = iterative_train(sparse_model, config, trainloader, testloader, device)
    trained_model = remove_pruning_from_model(sparse_model, config)
    summary(trained_model, (3,32,32))
    #torch.save(trained_model.state_dict(), "test_model.pt")
    #torch.save(trained_model, "test_model_full.pth")
    print("TESTING FINAL MODEL")
    test(trained_model, testloader, device)

