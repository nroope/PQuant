import os
os.environ["KERAS_BACKEND"] = "torch"
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import tqdm

from sparse_layers import  get_layer_keep_ratio, get_model_losses, add_pruning_to_model, call_post_epoch_function
from parser import get_parser
from data import get_cifar10_data
import keras_core as keras
keras.backend.set_image_data_format('channels_first')

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


def train(model, config, trainloader, testloader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=config.momentum, lr=config.lr, weight_decay=config.l2_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            loss += losses
            loss.backward()
            optimizer.step()

        test(model, testloader, device)
        call_post_epoch_function(model, epoch)
        scheduler.step()
    print('Finished Training')
    return model


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet34()
    sparse_model = add_pruning_to_model(model, config)
    sparse_model = sparse_model.to(device)
    from torchsummary import summary
    summary(sparse_model, (3,32,32))
    trainloader, testloader = get_cifar10_data(config.batch_size)
    trained_sparse_model = train(sparse_model, config, trainloader, testloader, device)

