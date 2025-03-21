import torch.nn as nn
import torch
import torch.nn.functional as F
import glob
import numpy as np
from tqdm import tqdm
from pquant.core.compressed_layers import get_model_losses, get_layer_keep_ratio

class SmartPixelModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sep_conv = depthwise_separable_conv(input_channels=2, output_channels=5)
        self.conv = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        self.avg = nn.AvgPool2d(3)
        self.flat = nn.Flatten(1)
        self.dense1 = nn.Linear(90, 16)
        self.dense2 = nn.Linear(16,16)
        self.dense3 = nn.Linear(16, 14)

    def forward(self, x):
        x = self.sep_conv(x)
        x = F.tanh(x)
        x = self.conv(x)
        x = F.tanh(x)
        
        x = self.avg(x)
        x = F.tanh(x)
    
        x = self.flat(x)
        x = self.dense1(x)
        x = F.tanh(x)
        x = self.dense2(x)
        x = F.tanh(x)
        x = self.dense3(x)
        return x

class depthwise_separable_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=0, groups=input_channels, bias=False)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# custom loss function
def custom_loss(p_base, y, minval=1e-9, maxval=1e9, scale = 512):
    
    p = p_base
    
    mu = p[:, 0:8:2]
    
    # creating each matrix element in 4x4
    Mdia = minval + torch.maximum(p[:, 1:8:2], torch.tensor(0.0))
    Mcov = p[:,8:]
    # placeholder zero element
    zeros = torch.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = torch.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = torch.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = torch.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = torch.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = torch.stack([row1,row2,row3,row4])
    scale_tril = scale_tril.permute((2,0,1))
    
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc = mu, scale_tril = scale_tril) 

    likelihood = torch.exp(dist.log_prob(y))
    likelihood = torch.clamp(likelihood,torch.tensor(minval).cuda(), torch.tensor(maxval).cuda())

    NLL = -1*torch.log(likelihood)

    NLL = torch.sum(NLL).cuda()
    return NLL


def get_smartpixel_data_and_model(data=True):
    model = SmartPixelModel().to(device="cuda")
    loss_func = custom_loss
    training_dataset_files = glob.glob("../data/smartpixel/train/*")

    if not data:
        return model, loss_func
    import random
    random.shuffle(training_dataset_files)
    tdata = []
    vdata = []
    print(len(training_dataset_files))
    for f in tqdm(training_dataset_files):
        data = np.load(f)
        inputs = data["inputs"].astype(np.float16)
        target = data["target"].astype(np.float16)
        tdata.append((inputs, target))
    
    validation_dataset_files = glob.glob("../data/smartpixel/validation/*")
    for f in tqdm(validation_dataset_files):
        data = np.load(f)
        inputs = data["inputs"].astype(np.float16)
        target = data["target"].astype(np.float16)
        vdata.append((inputs, target))
        
    return model, tdata, vdata, loss_func


## Training and validation ##

def train_smartpixel(model, train_data, optimizer, device, epoch, writer=None, *args, **kwargs):
    import random
    random.shuffle(train_data)
    for training_file in tqdm(train_data):
        inputs, target = training_file
        inputs = torch.tensor(inputs).to(dtype=torch.float32, device=device)
        target = torch.tensor(target).to(dtype=torch.float32, device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = custom_loss(outputs, target)
        losses = get_model_losses(model, torch.tensor(0.).to(device))
        loss += losses
        loss.backward()
        optimizer.step()
    if writer is not None:
        writer.add_scalar("training_loss", loss.item(), epoch)


def validate_smartpixel(model, validation_data, device, epoch, writer=None, save_outputs=False, *args, **kwargs):
    complete_truth = None
    p_test = None
    for training_file in tqdm(validation_data):
        inputs, target = training_file
        inputs = torch.tensor(inputs).to(dtype=torch.float32, device=device)
        target = torch.tensor(target).to(dtype=torch.float32, device=device)

        inputs = inputs.to(device)
        target = target.to(device)

        outputs = model(inputs)
        loss = custom_loss(outputs, target)
        losses = get_model_losses(model, torch.tensor(0.).to(device))
        loss += losses
        if save_outputs:
            if p_test is None:
                complete_truth = target.cpu().numpy()
                p_test = outputs.detach().cpu().numpy()
            else:
                p_test = np.concatenate((p_test, outputs.detach().cpu().numpy()), axis=0)
                complete_truth = np.concatenate((complete_truth, target.cpu().numpy()), axis=0)

    if save_outputs:
        np.savez("complete_truth.npz", complete_truth=complete_truth)
        np.savez("p_test.npz", p_test=p_test)
    
    ratio = get_layer_keep_ratio(model)
    if writer is not None:
        writer.add_scalar("validation_remaining_weights", ratio, epoch)
        writer.add_scalar("validation_loss", loss.item(), epoch)
