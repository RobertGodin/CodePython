
# -*- coding: utf-8 -*-
"""
Exemple simple de MNIST avec PyTorch
Exemple avec nn.Conv2d et F.cross_entropy
Production de métriques avec graphes
Fonction entrainer
"""
import torch
torch.manual_seed(0) # Pour résultats reproductibles

# Fonction J d'entropie croisée
import torch.nn.functional as F
fonction_cout = F.cross_entropy

def taux_bonnes_predictions(lot_Y_predictions, lot_Y):
    predictions_categorie = torch.argmax(lot_Y_predictions, dim=1)
    return (predictions_categorie == lot_Y).float().mean()

from torch import nn
# Définition de l'architecture du RNA

class BlocResSimple(nn.Module):
    def __init__(self, nb_f):
        super().__init__()
        self.convolution = nn.Conv2d(nb_f, nb_f, kernel_size=3, stride=1, padding=1) 
        self.normalisation = nn.BatchNorm2d(nb_f)    

    def forward(self, lot_X):
        lot_Y_predictions = self.convolution(lot_X)
        lot_Y_predictions = self.normalisation(lot_Y_predictions)
        lot_Y_predictions = F.relu(lot_Y_predictions)
        return lot_Y_predictions+lot_X     

class ResBlock(nn.Module):
  def __init__(self, nb_f):
    super().__init__()
    
    self.conv1 = nn.Conv2d(nb_f, nb_f, kernel_size=3, stride=1, padding=1, bias=False) 
    self.bn1 = nn.BatchNorm2d(nb_f)    
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(nb_f, nb_f, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(nb_f)    

  def forward(self, lot_X):
    lot_Y_predictions = self.conv1(lot_X)
    lot_Y_predictions = self.bn1(lot_Y_predictions)
    lot_Y_predictions = self.relu(lot_Y_predictions)
    lot_Y_predictions = self.conv2(lot_Y_predictions)
    lot_Y_predictions = self.bn2(lot_Y_predictions)
    lot_Y_predictions = lot_Y_predictions + lot_X
    lot_Y_predictions = self.relu(lot_Y_predictions)
    return lot_Y_predictions 
   

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

mod_res_v1 = nn.Sequential(
    nn.Conv2d(1,16,3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    ResBlock(16),    
    nn.Conv2d(16,8,3),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    ResBlock(8),
    nn.Flatten(),
    nn.Linear(24*24*8,10)
)

modele = nn.Sequential(
    nn.Conv2d(1,16,3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    BlocResSimple(16),
    BlocResSimple(16),
    BlocResSimple(16),
    nn.Flatten(),
    nn.Linear(26*26*16,10)
)


mod = nn.Sequential(
    nn.Conv2d(1,32,3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    
    
    nn.MaxPool2d(2),
    nn.Conv2d(32,16,3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(6*6*16,10)
)

from torch import optim
optimiseur = optim.SGD(modele.parameters(), lr=0.01)

# Chargement des données de MNIST
import pickle, gzip
fichier_donnees = gzip.open(r"mnist.pkl.gz", 'rb')
((donnees_ent_X, donnees_ent_Y),(donnees_valid_X, donnees_valid_Y),(donnees_test_X,donnees_test_Y)) = pickle.load(fichier_donnees, encoding="latin-1")
fichier_donnees.close()

# Conversion des données en type toch.Tensor
import torch
donnees_ent_X,donnees_ent_Y,donnees_valid_X,donnees_valid_Y,donnees_test_X,donnees_test_Y  = map(torch.tensor,
    (donnees_ent_X.reshape((-1,1,28,28)),donnees_ent_Y,donnees_valid_X.reshape((-1,1,28,28)),donnees_valid_Y,donnees_test_X.reshape((-1,1,28,28)),donnees_test_Y))

# Création des objets DataLoader pour itérer par lot
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
ds_ent = TensorDataset(donnees_ent_X, donnees_ent_Y)
dl_ent = DataLoader(ds_ent, batch_size=100, shuffle=True)
ds_valid = TensorDataset(donnees_valid_X,donnees_valid_Y)
dl_valid = DataLoader(ds_valid, batch_size=100)

def entrainer(modele, dl_ent, dl_valid, optimiseur, nb_epochs=10):

    # Listes pour les métriques par epoch
    liste_cout_moyen_ent = []
    liste_taux_moyen_ent = []
    liste_cout_moyen_valid = []
    liste_taux_moyen_valid = []
    
    # Boucle d'apprentissage
    for epoch in range(nb_epochs):
        cout_total_ent = 0 # pour cumuler les couts par mini-lot
        taux_bonnes_predictions_ent = 0 # pour cumuler les taux par mini-lot
        modele.train() # Pour certains types de couches (nn.BatchNorm2d, nn.Dropout, ...)
        
        # Boucle d'apprentissage par mini-lot pour une epoch
        for lot_X, lot_Y in dl_ent:
            optimiseur.zero_grad() # Remettre les dérivées à zéro
            lot_Y_predictions = modele(lot_X) # Appel de la méthode forward
            cout = fonction_cout(lot_Y_predictions, lot_Y)
            cout.backward() # Calcul des gradiants par rétropropagation
            with torch.no_grad():
                cout_total_ent +=cout
                taux_bonnes_predictions_ent += taux_bonnes_predictions(lot_Y_predictions, lot_Y)
            optimiseur.step() # Mise à jour des paramètres
        # Calculer les moyennes par mini-lot
        with torch.no_grad():
            cout_moyen_ent = cout_total_ent/len(dl_ent)
            taux_moyen_ent = taux_bonnes_predictions_ent/len(dl_ent)
       
        modele.eval() # Pour certains types de couches (nn.BatchNorm2d, nn.Dropout, ...)
        with torch.no_grad():
            cout_valid = sum(fonction_cout(modele(lot_valid_X), lot_valid_Y) for lot_valid_X, lot_valid_Y in dl_valid)
            taux_bons_valid = sum(taux_bonnes_predictions(modele(lot_valid_X), lot_valid_Y) for lot_valid_X, lot_valid_Y in dl_valid)
        cout_moyen_valid = cout_valid/len(dl_valid)
        taux_moyen_valid = taux_bons_valid/len(dl_valid)
        print(f'-------- > epoch {epoch+1}:  coût moyen entraînement = {cout_moyen_ent}')
        print(f'-------- > epoch {epoch+1}:  taux moyen entraînement = {taux_moyen_ent}')
        print(f'-------- > epoch {epoch+1}:  coût moyen validation = {cout_moyen_valid}')
        print(f'-------- > epoch {epoch+1}:  taux moyen validation = {taux_moyen_valid}')
    
        liste_cout_moyen_ent.append(cout_moyen_ent)
        liste_taux_moyen_ent.append(taux_moyen_ent)
        liste_cout_moyen_valid.append(cout_moyen_valid)
        liste_taux_moyen_valid.append(taux_moyen_valid)
    
    # Affichage du graphique d'évolution des métriques par epoch
    import numpy as np
    import matplotlib.pyplot as plt
    plt.plot(np.arange(0,nb_epochs),liste_cout_moyen_ent,label='Erreur entraînement')
    plt.plot(np.arange(0,nb_epochs),liste_cout_moyen_valid,label='Erreur validation')
    plt.title("Evolution du coût")
    plt.xlabel('epoch')
    plt.ylabel('moyenne par observation')
    plt.legend(loc='upper center')
    plt.show()
        
    plt.plot(np.arange(0,nb_epochs),liste_taux_moyen_ent,label='Taux bonnes réponses entraînement')
    plt.plot(np.arange(0,nb_epochs),liste_taux_moyen_valid,label='Taux bonnes réponses validation')
    plt.title("Evolution du taux")
    plt.xlabel('epoch')
    plt.ylabel('moyenne par observation')
    plt.legend(loc='upper center')
    plt.show()

entrainer(modele, dl_ent, dl_valid, optimiseur, nb_epochs=30)