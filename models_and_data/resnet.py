"""Code taken from https://github.com/akamaster/pytorch_resnet_cifar10"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import tqdm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16
from pquant.core.compressed_layers import get_model_losses, get_layer_keep_ratio

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


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
    if writer is not None:
        writer.add_scalar("train_output_loss", loss.item(), epoch)
        writer.add_scalar("train_sparse_loss", losses, epoch)

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
            ratio = get_layer_keep_ratio(model)
            print(f'Accuracy: {100 * correct / total}%, remaining_weights: {ratio * 100}%')
        if writer is not None:
            writer.add_scalar("validation_output_loss", loss.item(), epoch)
            writer.add_scalar("validation_sparse_loss", losses, epoch)
            writer.add_scalar("validation_acc", correct / total, epoch)
            writer.add_scalar("validation_remaining_weights", ratio, epoch)



def get_resnet_model(config, device):
    if config["model"] == "resnet18":
        model = resnet18().to(device)
    elif config["model"] == "resnet34":
        model = resnet34().to(device)
    elif config["model"] == "resnet50":
        model = resnet50().to(device)
    elif config["model"] == "resnet101":
        model = resnet101().to(device)
    elif config["model"] == "resnet152":
        model = resnet152().to(device)
    elif config["model"] == "resnet20":
        model = resnet20().to(device)
    elif config["model"] == "resnet32":
        model = resnet32().to(device)
    elif config["model"] == "resnet44":
        model = resnet44().to(device)
    elif config["model"] == "resnet56":
        model = resnet56().to(device)
    elif config["model"] == "resnet110":
        model = resnet110().to(device)
    elif config["model"] == "resnet1202":
        model = resnet1202().to(device)
    elif config["model"] == "vgg16":
        model = vgg16().to(device)
    return model



if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()