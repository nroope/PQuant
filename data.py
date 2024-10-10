import torch
import torchvision
import torchvision.transforms as transforms
import keras_core as keras
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

def get_cifar10_data(batch_size):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    return trainloader, testloader
    

class JetTaggingDataSet(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        self.x = data["data"]
        self.y = data["target"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx].astype(np.float32), self.y[idx].astype(np.float32)


def get_jet_tagging_data(batch_size):

    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X, y = data['data'], data['target']
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = keras.utils.to_categorical(y, 5)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    train_dataset = JetTaggingDataSet({"data": X_train_val, "target": y_train_val})
    test_dataset = JetTaggingDataSet({"data": X_test, "target": y_test})
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    return train_dataloader, test_dataloader

    