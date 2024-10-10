import os
os.environ["KERAS_BACKEND"] = "torch"
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import tqdm

from sparse_layers import  get_layer_keep_ratio, get_model_losses, add_pruning_to_model, call_post_epoch_function, call_post_round_function, call_save_weights_function, call_rewind_weights_function, call_pre_finetune_function
from parser import get_parser
from data import get_cifar10_data
import keras_core as keras
keras.backend.set_image_data_format('channels_first')



def post_round_functions(model, config, round):
        if config.rewind == "round":
            call_rewind_weights_function(model)
        elif config.rewind == "post-ticket-search" and round == config.rounds - 1:
            call_rewind_weights_function(model)
        if round < config.rounds - 1:
            call_post_round_function(model)


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
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        ratios = get_layer_keep_ratio(model, torch.Tensor([1.0]).to(device))
        print("Remaining weights", torch.mean(ratios))


def iterative_train(model, config, trainloader, testloader, device, criterion=None, optimizer=None, scheduler=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), momentum=config.momentum, lr=config.lr, weight_decay=config.l2_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[56,71], gamma=0.1)
    for r in range(config.rounds):
        train(model, config, trainloader, testloader, device, criterion, optimizer, scheduler, r)
        post_round_functions(model, config, r)
    if config.fine_tune:
        call_pre_finetune_function(model)
        train(model, config, trainloader, testloader, device, criterion, optimizer, scheduler, config.rounds)

def train(model, config, trainloader, testloader, device, criterion=None, optimizer=None, scheduler=None, round=0):
    #lambda_reg = 0.0001
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        if round == 0 and config.save_weights_epoch == epoch:
            call_save_weights_function(model)

        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            loss += losses
            #L1-regularization
            #for n, v in model.named_parameters():
            #    if "pruning_layer" not in n:
            #        loss += lambda_reg * v.abs().sum()
            loss.backward()
            #for n, v in model.named_parameters():
                #print(n, v.grad)
            optimizer.step()

        test(model, testloader, device)
        call_post_epoch_function(model, epoch, config.epochs)
        if scheduler is not None:
            scheduler.step()
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

