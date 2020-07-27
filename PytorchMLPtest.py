# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting beautiful graphs

# train test split from sklearn
from sklearn.model_selection import train_test_split

# Import Torch 
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import torchvision.datasets as dset
# from torch.utils.data import SubsetRandomSampler
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# What's in the current directory?
import os
print(os.listdir("/"))


t = transforms.Compose([transforms.ToTensor()])

dl_train = DataLoader( torchvision.datasets.MNIST('/data/mnist', download=True, train=True, transform=t), 
                batch_size=100, shuffle=True)
dl_valid = DataLoader( torchvision.datasets.MNIST('/data/mnist', download=True, train=False, transform=t), 
                batch_size=100,shuffle=True)


print('total trainning batch number:',len(dl_train))
print('==>>> total testing batch number:',len(dl_valid))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784,30)
        self.linear2 = nn.Linear(30,10)
    
    def forward(self,X):
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return F.log_softmax(X, dim=1)
 
mlp = MLP()
print(mlp)

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data[0], float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
                
fit(mlp, dl_train)

def evaluate(model):
#model = mlp
    correct = 0 
    for test_imgs, test_labels in dl_valid:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(dl_valid)*BATCH_SIZE)))
evaluate(mlp)

