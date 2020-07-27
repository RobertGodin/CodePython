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

class ResBlock(nn.Module):
  def __init__(self, nf):
    super().__init__()
    
    self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False) 
    self.bn1 = nn.BatchNorm2d(nf)    
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(nf)    

  def forward(self, x):
    o = self.conv1(x)
    o = self.bn1(o)
    o = self.relu(o)
    o = self.conv2(o)
    o = self.bn2(o)
    o = o + x
    o = self.relu(o)
    return o 
   

# kernel of 3x3, stride of 2 and padding 1
def convk3s2p1(ni, nf):
    return nn.Sequential(
        nn.Conv2d(ni, nf, 3,2,1, bias=False),
        nn.BatchNorm2d(nf),
        nn.ReLU()
    )
        
model=nn.Sequential(
    convk3s2p1(1,8),
    ResBlock(8),
    convk3s2p1(8,16),
    ResBlock(16),
    convk3s2p1(16,32),
    ResBlock(32),
    convk3s2p1(32,16),
    convk3s2p1(16,10),
    nn.Flatten()
)

# read 85 as BS
input=(torch.randn(85, 1, 24,24))
output=model(input)
print(output.shape)
model

optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

epochs=10
losses=[]

total_steps=len(dl_train)*epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=0.5, 
                                                steps_per_epoch=len(dl_train),
                                                epochs=epochs)

from sklearn.metrics import accuracy_score, f1_score

for epoch in range(1,epochs+1):
    print(f"Epoch {epoch}")

    # TRAINING PHASE
    model.train()

   
    ect=0 # epoch correct test
    ecv=0 # epoch correct validation
    ett=0 # len of epoch train examples
    etv=0 # len of epoch validation examples   

    for i, (input,target) in enumerate(dl_train):
                
        if(i%10==0): print(i, end=" ")
        optimizer.zero_grad()
                
        output = model(input)
        ect+= accuracy_score(output.argmax(dim=-1).cpu(), target.cpu(), normalize=False)
                

        loss = loss_fn(output, target) # one batch loss        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step() 
        ett+=len(target)    
        

    # VALIDATION PHASE
    model.eval() 
    with torch.no_grad(): #gradients should not eval        
        for j, (input,target) in enumerate(dl_valid):

            output = model(input)
            ecv+= accuracy_score(output.argmax(dim=-1).cpu(), target.cpu(), normalize=False)
            etv+=len(target)            

    print("")
    print("Epoch training accuracy" , ect/ ett)
    # experiment.log_metric("Epoch training accuracy", ect/ ett)
    print("Epoch valid accuracy" , ecv/ etv)
    # experiment.log_metric("Epoch valid accuracy", ecv/ etv)

