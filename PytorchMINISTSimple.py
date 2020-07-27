# -*- coding: utf-8 -*-
"""
Exemple simple de MNIST avec PyTorch
"""

# Chargement des données de MNIST
import pickle, gzip
fichier_donnees = gzip.open(r"mnist.pkl.gz", 'rb')
((donnees_ent_X, donnees_ent_Y),(donnees_valid_X, donnees_valid_Y),(donnees_test_X,donnees_test_Y)) = pickle.load(fichier_donnees, encoding="latin-1")
fichier_donnees.close()

# Conversion des données en type toch.Tensor
import torch
donnees_ent_X, donnees_ent_Y, donnees_test_X,donnees_test_Y = map(torch.tensor, (donnees_ent_X, donnees_ent_Y, donnees_test_X,donnees_test_Y))

# Création des objets DataLoader pour itérer par lot
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
ds_ent = TensorDataset(donnees_ent_X, donnees_ent_Y)
dl_ent = DataLoader(ds_ent, batch_size=100, shuffle=True)
ds_test = TensorDataset(donnees_test_X,donnees_test_Y)
dl_test = DataLoader(ds_test, batch_size=100)

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
            lot_X = F.relu(self.couche_lineaire1(lot_X))
            return self.couche_lineaire2(lot_X)
modele = RNASimple()
    
from torch import optim
optimiseur = optim.SGD(modele.parameters(), lr=0.003)

graph_cout=True

if graph_cout :
    liste_cout_moyen_ent = []
    liste_ok_ent = []
    liste_cout_moyen_test = []
    liste_ok_test = []

nb_epochs = 30
# Boucle d'apprentissage
for epoch in range(nb_epochs):
    cout_total_ent = 0
    modele.train() # Pour certains types de couches (nn.BatchNorm2d, nn.Dropout, ...)
    
    # Boucle d'apprentissage par mini-lot pour une epoch
    for lot_X, lot_Y in dl_ent:
        lot_Y_predictions = modele(lot_X) # Appel de la méthode forward
        cout = fonction_cout(lot_Y_predictions, lot_Y)
        cout_total_ent +=cout

        cout.backward() # Calcul des dérivées par rétropropagation
        optimiseur.step() # Mise à jour des paramètres
        optimiseur.zero_grad() # Remettre les dérivées à zéro
        
    cout_moyen_ent = cout_total_ent/len(dl_ent)
    print(f'-------- > epoch {epoch+1}:  coût moyen entraînement = {cout_moyen_ent}')
   
    modele.eval() # Pour certains types de couches (nn.BatchNorm2d, nn.Dropout, ...)
    with torch.no_grad():
        cout_test = sum(fonction_cout(modele(lot_test_X), lot_test_Y) for lot_test_X, lot_test_Y in dl_test)
    cout_moyen_test = cout_test / len(dl_test)
    print(f'-------- > epoch {epoch+1}:  coût moyen test = {cout_moyen_test}')
    
    if graph_cout:
        liste_cout_moyen_ent.append(cout_moyen_ent)
        liste_cout_moyen_test.append(cout_moyen_test)

# Affichage du graphique d'évolution de l'erreur quadratique
import numpy as np
import matplotlib.pyplot as plt
if graph_cout:
    plt.plot(np.arange(0,nb_epochs),liste_cout_moyen_ent,label='Erreur entraînement')
    plt.plot(np.arange(0,nb_epochs),liste_cout_moyen_test,label='Erreur test')
    plt.title("Evolution du coût")
    plt.xlabel('epoch')
    plt.ylabel('moyenne par observation')
    plt.legend(loc='upper center')
    plt.show()
