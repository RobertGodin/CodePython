# -*- coding: utf-8 -*-
"""
Exemple de Réseau de Neurone Récurrent avec Pytorch RNN

"""
import torch
from torch import nn
import numpy as np
torch.manual_seed(0) # Pour résultats reproductibles

# Préparer les données d'entrainement
liste_chaines = ['wifi','wiki','sifi','kiwi']
ensemble_caracteres = sorted(list(set(''.join(liste_chaines))))
dict_int_car = dict(enumerate(ensemble_caracteres))
dict_car_int = {char: ind for ind, char in dict_int_car.items()}
print("Correspondances des caractères en entiers:",dict_car_int)
taille_max_chaine = len(max(liste_chaines, key=len))
print("Taille maximale de chaine :",taille_max_chaine)

# Niveler les tailles des chaines pour simplifier le traitement
for i in range(len(liste_chaines)):
    while len(liste_chaines[i])<taille_max_chaine:
        liste_chaines[i] += ' '
        
mini_lot_sequence_X = [] # Mini_lot de séquences X pour l'entraînement
mini_lot_sequence_Y = [] # Mini_lot de séquences cibles X pour l'entraînement

for i in range(len(liste_chaines)):
    mini_lot_sequence_X.append(liste_chaines[i][:-1]) # Supprimer dernier caractère de la sequence X
    mini_lot_sequence_Y.append(liste_chaines[i][1:])  # Supprimer premier caractère de la sequence Y
print("Mini-lot des séquences de caractères X")
print(mini_lot_sequence_X)
print("Mini-lot des séquences de caractères Y")
print(mini_lot_sequence_Y)

for i in range(len(liste_chaines)): # Conversion des caractères en entiers
    mini_lot_sequence_X[i] = [dict_car_int[character] for character in mini_lot_sequence_X[i]]
    mini_lot_sequence_Y[i] = [dict_car_int[character] for character in mini_lot_sequence_Y[i]]
print("Mini-lot des séquences X sous forme d'entiers")
print(mini_lot_sequence_X)
print("Mini-lot des séquences Y sous forme d'entiers")
print(mini_lot_sequence_Y)

#Coder les entiers de Y en bitmap pour l'entrainement
taille_dictionnaire = len(dict_car_int)    
taille_sequence = taille_max_chaine - 1
taille_mini_lot = len(liste_chaines)
mini_lot_sequence_X_bitmap = np.zeros((taille_mini_lot, taille_sequence, taille_dictionnaire), dtype=np.float32)
for indice_lot in range(taille_mini_lot):
    for indice_sequence in range(taille_sequence):
        mini_lot_sequence_X_bitmap[indice_lot, indice_sequence, mini_lot_sequence_X[indice_lot][indice_sequence]] = 1
print("Mini-lot des séquences X sous forme de bitmaps (encodage one-hot)")
print("Forme de X: {} --> (taille mini lot, taille sequence, taille bitmap)".format(mini_lot_sequence_X_bitmap.shape))
print(mini_lot_sequence_X_bitmap)

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
        H = torch.zeros(self.nb_couches_RNR, taille_mini_lot, self.taille_H).to(device)
        lot_Ht, H = self.rnn(lot_X, H) # lot_Ht : (taille_mini_lot,taille_sequence,taille_H)
        # Applatir (taille_mini_lot,taille_sequence) pour la couche dense qui suit
        lot_Ht = lot_Ht.contiguous().view(-1, self.taille_H)  # lot_Ht : (taille_mini_lot*taille_sequence,taille_H)
        lot_Yt = self.fc(lot_Ht) # lot_Yt : (taille_mini_lot*taille_sequence,taille_Y)
        
        return lot_Yt, H
    
    def init_H(self, taille_mini_lot):
        H = torch.zeros(self.nb_couches_RNR, taille_mini_lot, self.taille_H).to(device)
        return H

modele = Modele(taille_X=taille_dictionnaire, taille_Y=taille_dictionnaire, taille_H=6, nb_couches_RNR=1)
modele = modele.to(device)

n_epochs = 100
taux=0.01
fonction_cout = nn.CrossEntropyLoss()
optimizeur = torch.optim.Adam(modele.parameters(), lr=taux)

# Entraînement du RNR
mini_lot_sequence_X_bitmap = mini_lot_sequence_X_bitmap.to(device)
for epoch in range(1, n_epochs + 1):
    optimizeur.zero_grad()
    lot_Yt, H = modele(mini_lot_sequence_X_bitmap)
    lot_Yt = lot_Yt.to(device)
    mini_lot_sequence_Y = mini_lot_sequence_Y.to(device)
    cout = fonction_cout(lot_Yt, mini_lot_sequence_Y.view(-1).long())
    cout.backward()
    optimizeur.step()
    if epoch%10 == 0:
        print(f'-------- > epoch {epoch}:  coût = {cout}')
        
def prediction(modele, ensemble_caracteres):
    """ Retourner le prochain caratère de la séquence ensemble_caracteres selon le modèle"""
    # Transformer l'ensemble en un mini-lot de taille 1 avec le format approprié (1,taille_seq,bitmap)
    ensemble_caracteres = np.array([[dict_car_int[c] for c in ensemble_caracteres]])
    taille_sequence = ensemble_caracteres.shape[1]
    mini_lot_sequence_X_bitmap = np.zeros((1, taille_sequence, taille_dictionnaire), dtype=np.float32)
    for indice_sequence in range(taille_sequence):
        mini_lot_sequence_X_bitmap[0, indice_sequence, ensemble_caracteres[0][indice_sequence]] = 1
    mini_lot_sequence_X_bitmap = torch.from_numpy(mini_lot_sequence_X_bitmap)
    mini_lot_sequence_X_bitmap = mini_lot_sequence_X_bitmap.to(device)
    Yt, H = modele(mini_lot_sequence_X_bitmap)
    print("Un seul lot X<1>, X<2>, ..., X<t>:",mini_lot_sequence_X_bitmap)
    print("Prédiction Y<1>,Y<2>, ... ,Y<t>:",Yt)
    softmax_dernier_Yt = nn.functional.softmax(Yt[-1], dim=0).data
    print("Softmax du dernier Yt:",softmax_dernier_Yt)
    indice_probabilite_maximale = torch.max(softmax_dernier_Yt, dim=0)[1].item()
    
    return dict_int_car[indice_probabilite_maximale]

def echantillon(modele, taille_resultat, prefixe='w'):
    """ Compléter le préfixe par échantillonnage du modèle un caractère à la fois"""
    modele.eval()
    ensemble_caracteres = [caractere for caractere in prefixe]
    taille_restante = taille_resultat - len(ensemble_caracteres)
    for i in range(taille_restante):
        print("Prédiction itération: ", i)
        caractere_prediction = prediction(modele, ensemble_caracteres)
        print("Caractère prédit: ", caractere_prediction)
        ensemble_caracteres.append(caractere_prediction)
    return ''.join(ensemble_caracteres)

print(echantillon(modele,4,'w'))


