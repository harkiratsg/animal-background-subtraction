import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data_utils

class PatchDataset(Dataset):
    def __init__(self, img_patches, p_in_bounds, gpu=False):
        self.img_patches = torch.tensor(img_patches).float()
        self.p_within_bounds = torch.tensor(p_in_bounds).float()

        if gpu:
            self.img_patches = self.img_patches.cuda()
            self.p_within_bounds = self.p_within_bounds.cuda()

    def __len__(self):
        return self.img_patches.shape[0]

    def __getitem__(self, idx):
        return self.img_patches[idx], self.p_within_bounds[idx]

class PatchClassifier(nn.Module):
    def __init__(self, width, height, layer_dims, k):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, layer_dims[0], 5, padding=2)
        self.conv_layers = nn.ModuleList([nn.Conv2d(layer_dims[i], layer_dims[i+1], k, padding=k//2) for i in range(len(layer_dims)-1)])

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(layer_dims[-1] * (width // (2 ** len(layer_dims))) * (height // (2 ** len(layer_dims))), 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def saliency_map(self, x, layer_i):
        x = self.pool(F.relu(self.conv1(x)))

        for i, layer in enumerate(self.conv_layers):
            if i == layer_i:
                break
            x = self.pool(F.relu(layer(x)))
        
        return x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        for i, layer in enumerate(self.conv_layers):
            x = self.pool(F.relu(layer(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.sigmoid(x)

def accuracy(y, y_pred):
    """ Compute accuracy metric """
    with torch.no_grad():
        y_bin = [int(v) for v in y.cpu().numpy()]
        y_pred_bin = [int(round(v)) for v in y_pred.cpu().numpy()]

        acc = sum([1 if true == pred else 0 for true, pred in zip(y_bin, y_pred_bin)]) / len(y_bin)

    return acc
    
def train_patch_classifier(model, n_epochs, train_dataloader):
    """ Train a patch classifier model given the dataloader """
    train_x = torch.stack([x for x, y in train_dataloader.dataset])
    train_y = torch.stack([y for x, y in train_dataloader.dataset])

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(n_epochs):
        epoch_losses = []
        for x, y in train_dataloader:
            y_pred = model(x)
            loss = criterion(y_pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
        acc = accuracy(train_y[:1000], model(train_x[:1000]).flatten())
        print(f'EPOCH {epoch} - loss: {sum(epoch_losses) / len(epoch_losses)} acc: {acc}')


def get_trained_patch_model(img_patches, p_in_bounds, model_config, n_epochs, batch_size, gpu=False):
    """ Get trained patch model given training config """
    model = PatchClassifier(**model_config)
    if gpu:
        model = model.cuda()

    train_dataloader = DataLoader(PatchDataset(*data_utils.prepare_data(img_patches, p_in_bounds, binary=True), gpu=gpu), batch_size=batch_size, shuffle=True)
    train_patch_classifier(model, n_epochs, train_dataloader)

    return model