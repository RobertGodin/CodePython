# -*- coding: utf-8 -*-
"""
Exemple de Réseau de Neurone Récurrent avec Pytorch RNN

"""
import torch
from torch import nn
import numpy as np

# Préparer les données d'entrainement
liste_chaines = ['wifi','wiki','witz','kiwi']
ensemble_caracteres = sorted(list(set(''.join(liste_chaines))))
dict_int_car = dict(enumerate(ensemble_caracteres))
dict_car_int = {char: ind for ind, char in dict_int_car.items()}
print("Correspondanes des caractères en eniers:",dict_car_int)
taille_max_chaine = len(max(liste_chaines, key=len))
print("Taille maximale de chaine :",taille_max_chaine)

# Niveler les tailles des chaines pour simplifier le traitement
for i in range(len(liste_chaines)):
    while len(liste_chaines[i])<taille_max_chaine:
        liste_chaines[i] += ' '
        
mini_lot_sequence_X = [] # Mini_lot de séquences X pour l'entraînement
mini_lot_sequence_Y = [] # Mini_lot de séquences cibles X pour l'entraînement

print("Mini-lot des séquences de caractères")
for i in range(len(liste_chaines)):
    mini_lot_sequence_X.append(liste_chaines[i][:-1]) # Supprimer dernier caractère de la sequence X
    mini_lot_sequence_Y.append(liste_chaines[i][1:])  # Supprimer premier caractère de la sequence Y
    print("Sequence X: {} Sequence Y: {}".format(mini_lot_sequence_X[i], mini_lot_sequence_Y[i]))

print("Mini-lot des séquences sous forme d'entiers")
for i in range(len(liste_chaines)): # Conversion des caractères en entiers
    mini_lot_sequence_X[i] = [dict_car_int[character] for character in mini_lot_sequence_X[i]]
    mini_lot_sequence_Y[i] = [dict_car_int[character] for character in mini_lot_sequence_Y[i]]
    print("Sequence X: {} Sequence Y: {}".format(mini_lot_sequence_X[i], mini_lot_sequence_Y[i]))
    
taille_dictionnaire = len(dict_car_int)    
taille_sequence = taille_max_chaine - 1
taille_mini_lot = len(liste_chaines)

def one_hot_encode(mini_lot_sequence_X, taille_dictionnaire, taille_sequence, taille_mini_lot):
    #Coder les entiers en bitmap
    mini_lot_sequence_X_bitmap = np.zeros((taille_mini_lot, taille_sequence, taille_dictionnaire), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for indice_lot in range(taille_mini_lot):
        for indice_sequence in range(taille_sequence):
            mini_lot_sequence_X_bitmap[indice_lot, indice_sequence, mini_lot_sequence_X[indice_lot][indice_sequence]] = 1
    return mini_lot_sequence_X_bitmap

mini_lot_sequence_X_bitmap = one_hot_encode(mini_lot_sequence_X, taille_dictionnaire, taille_sequence, taille_mini_lot)
print("Forme de X: {} --> (taille mini lot, taille sequence, taille bitmap)".format(mini_lot_sequence_X_bitmap.shape))
mini_lot_sequence_X_bitmap = torch.from_numpy(mini_lot_sequence_X_bitmap)
mini_lot_sequence_Y = torch.Tensor(mini_lot_sequence_Y)

# Déterminer si un GPU est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Entrainement sur ',device)
    
class Modele(nn.Module):
    def __init__(self, taille_X, taille_Y, taille_H, nb_couches_RNR):
        super(Modele, self).__init__()
        self.taille_H = taille_H
        self.nb_couches_RNR = nb_couches_RNR
        self.rnn = nn.RNN(taille_X, taille_H, nb_couches_RNR, batch_first=True)
        self.fc = nn.Linear(taille_H, taille_Y)
    
    def forward(self, lot_X):
        """ lot_X : (taille_mini_lot, taille_sequence, taille_bitmap)"""
        taille_mini_lot = lot_X.size(0)
        H = self.init_H(taille_mini_lot)
        print("Valeur initiale de H")
        print(H)
        Yt, H = self.rnn(lot_X, H)
        print("Valeur finale de H ")
        print(H) 
        print("Valeur de Yt")
        print(Yt)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        Yt = Yt.contiguous().view(-1, self.taille_H)
        Yt = self.fc(Yt)
        
        return Yt, H
    
    def init_H(self, taille_mini_lot):
        H = torch.zeros(self.nb_couches_RNR, taille_mini_lot, self.taille_H).to(device)
        return H

modele = Modele(taille_X=taille_dictionnaire, taille_Y=taille_dictionnaire, taille_H=5, nb_couches_RNR=1)
modele = modele.to(device)

n_epochs = 10
lr=0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modele.parameters(), lr=lr)

# Entraînement du RNR
mini_lot_sequence_X_bitmap = mini_lot_sequence_X_bitmap.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #mini_lot_sequence_X = mini_lot_sequence_X.to(device)
    output, H = modele(mini_lot_sequence_X_bitmap)
    output = output.to(device)
    mini_lot_sequence_Y = mini_lot_sequence_Y.to(device)
    loss = criterion(output, mini_lot_sequence_Y.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

def prediction(modele, ensemble_caracteres):
    """ Retourner le prochain caratère de la séquence ensemble_caracteres selon le modèle"""
    ensemble_caracteres = np.array([[dict_car_int[c] for c in ensemble_caracteres]])
    ensemble_caracteres = one_hot_encode(ensemble_caracteres, taille_dictionnaire, ensemble_caracteres.shape[1], 1)
    ensemble_caracteres = torch.from_numpy(ensemble_caracteres)
    ensemble_caracteres = ensemble_caracteres.to(device)
    Yt, H = modele(ensemble_caracteres)
    prob = nn.functional.softmax(Yt[-1], dim=0).data
    indice_probabilite_maximale = torch.max(prob, dim=0)[1].item()
    return dict_int_car[indice_probabilite_maximale], H

def echantillon(modele, taille_resultat, prefixe='w'):
    """ Compléter le préfixe par échantillonnage du modèle un caractère à la fois"""
    modele.eval()
    ensemble_caracteres = [caractere for caractere in prefixe]
    taille_restante = taille_resultat - len(ensemble_caracteres)
    # Now pass in the previous characters and get a new one
    for _ in range(taille_restante):
        caractere_prediction, H = prediction(modele, ensemble_caracteres)
        ensemble_caracteres.append(caractere_prediction)
    return ''.join(ensemble_caracteres)

print(echantillon(modele,4,'w'))


