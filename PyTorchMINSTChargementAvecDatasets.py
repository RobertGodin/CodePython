# -*- coding: utf-8 -*-
"""
Exemple simple de MNIST avec PyTorch
Exemple de nn.Module avec deux couches denses linéaires
Boucle d'apprentissage simple
"""
import torch
torch.manual_seed(0) # Pour résultats reproductibles

# Fonction J d'entropie croisée
import torch.nn.functional as F
fonction_cout = F.cross_entropy

from torch import nn
# Définition de l'architecture du RNA
class RNASimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche_lineaire1 = nn.Linear(784, 30)
        self.couche_lineaire2 = nn.Linear(30, 10)
    def forward(self, lot_X):
            lot_X = lot_X.view(lot_X.size()[0], -1)
            lot_X = F.relu(self.couche_lineaire1(lot_X))
            return self.couche_lineaire2(lot_X)
modele = RNASimple()
    
from torch import optim
optimiseur = optim.SGD(modele.parameters(), lr=0.05)

import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize = (3,3)) #define the image size

#transforming the PIL Image to tensors
ds_ent = torchvision.datasets.MNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
ds_test = torchvision.datasets.MNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())

#loading the training data from trainset
dl_ent = torch.utils.data.DataLoader(ds_ent, batch_size=100, shuffle = True)
#loading the test data from testset
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=100, shuffle=False)



def imshow(img):
     npimg = img.numpy() #convert the tensor to numpy for displaying the image
     #for displaying the image, shape of the image should be height * width * channels 
     plt.imshow(np.transpose(npimg, (1, 2, 0))) 
     plt.show()
     
#sneak peak into the train data

#iterating into the data
dataiter = iter(dl_ent)
images, labels = dataiter.next()

print(images.shape) #shape of all 4 images
print(images[1].shape) #shape of one image
print(labels[1].item()) #label number

imshow(torchvision.utils.make_grid(images))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5), #(N, 1, 28, 28) -> (N, 6, 24, 24)
            nn.Tanh(),
            nn.AvgPool2d(2, stride = 2), #(N, 6, 24, 24) -> (N, 6, 12, 12)
            
            nn.Conv2d(6, 16, kernel_size = 5), #(N, 6, 12, 12) -> (N, 6, 8, 8)
            nn.Tanh(),
            nn.AvgPool2d(2, stride = 2)) #(N, 6, 8, 8) -> (N, 16, 4, 4)
    
        self.fc_model = nn.Sequential(
            nn.Linear(256, 120), # (N, 256) -> (N, 120)
            nn.Tanh(),
            nn.Linear(120, 84), # (N, 120) -> (N, 84)
            nn.Tanh(),
            nn.Linear(84, 10))  # (N, 84)  -> (N, 10))
            
    def forward(self, x):
        #print(x.shape)
        x = self.cnn_model(x)
        #print(x.shape)
        #print(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_model(x)
        #print(x.shape)
        return x

nb_epochs = 10
# Boucle d'apprentissage
for epoch in range(nb_epochs):
    cout_total_ent = 0 
    
    # Boucle d'apprentissage par mini-lot pour une epoch
    for lot_X, lot_Y in dl_ent:
        optimiseur.zero_grad() # Remettre les dérivées à zéro
        lot_Y_predictions = modele(lot_X) # Appel de la méthode forward
        cout = fonction_cout(lot_Y_predictions, lot_Y)
        cout_total_ent +=cout
        cout.backward() # Calcul des gradiants par rétropropagation
        optimiseur.step() # Mise à jour des paramètres
        
    cout_moyen_ent = cout_total_ent/len(dl_ent)
    print(f'-------- > epoch {epoch+1}:  coût moyen entraînement = {cout_moyen_ent}')