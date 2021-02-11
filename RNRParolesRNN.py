# -*- coding: utf-8 -*-
"""
Réseau de neurones récurrent, modèle de langue par mot, pou paroles de chansons
Version avec couche vectorisation et RNN 
"""

import torch
from torch import nn
import pandas as pd
from collections import Counter

class DatasetParoles(torch.utils.data.Dataset):
    """ Créer un Dataset avec les paroles de la colonne Lyric du fichier 
    https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv
    taille_sequence : taille d'une séquence de mots pour le modèle de langue
    Le texte est découpé en séquences de la taille taille_sequence
    """
    def __init__(self,taille_sequence=6):
        self.taille_sequence = taille_sequence
        self.mots = self.charger_mots()
        self.mots_uniques = self.chercher_mots_uniques()

        self.index_a_mot = {index: mot for index, mot in enumerate(self.mots_uniques)}
        self.mot_a_index = {mot: index for index, mot in enumerate(self.mots_uniques)}

        self.mots_indexes = [self.mot_a_index[w] for w in self.mots]

    def charger_mots(self):
        dataframe_entrainment = pd.read_csv('lyrics-data.csv')
        text = dataframe_entrainment.iloc[0:10]['Lyric'].str.cat(sep=' ')
        return text.split(' ')

    def chercher_mots_uniques(self):
        frequence_mot = Counter(self.mots)
        return sorted(frequence_mot, key=frequence_mot.get, reverse=True)

    def __len__(self):
        return len(self.mots_indexes) - self.taille_sequence

    def __getitem__(self, index):
        return (
            torch.tensor(self.mots_indexes[index:index+self.taille_sequence]),
            torch.tensor(self.mots_indexes[index+1:index+self.taille_sequence+1]),
        )



class Modele(nn.Module):
    """Modèle de RNR avec une couche vectorisation, suivie d'une couche RNN et d'une couche linéaire"""
    def __init__(self, ds_paroles):
        super(Modele, self).__init__()
        self.taille_H_LSTM = 128
        self.taille_vectorisation_mots = 128
        self.nombre_couches_RNR = 1

        taille_vocabulaire = len(ds_paroles.mots_uniques)
        self.vectorisation_mots = nn.Embedding(num_embeddings=taille_vocabulaire,
            embedding_dim=self.taille_vectorisation_mots)
        self.lstm = nn.RNN(input_size=self.taille_H_LSTM,hidden_size=self.taille_H_LSTM,
            num_layers=self.nombre_couches_RNR,dropout=0.2)
        self.fc = nn.Linear(self.taille_H_LSTM, taille_vocabulaire)

    def forward(self, lot_X, etat_0):
        vectorisation = self.vectorisation_mots(lot_X)
        lot_Ht, etat = self.lstm(vectorisation, etat_0)
        lot_Yt = self.fc(lot_Ht)

        return lot_Yt, etat

    def initializer_etat(self, taille_sequence):
        return (torch.zeros(self.nombre_couches_RNR, taille_sequence, self.taille_H_LSTM))
    
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

def entrainer_RNR(ds_paroles, modele, taille_lot=32, epochs=10, taille_sequence=6):
    modele.train()
    dl_paroles = DataLoader(ds_paroles,batch_size=taille_lot)

    fonction_cout = nn.CrossEntropyLoss()
    optimizeur = optim.Adam(modele.parameters(), lr=0.001)

    for epoch in range(epochs):
        etat = modele.initializer_etat(taille_sequence)

        for lot, (lot_X, lot_Y) in enumerate(dl_paroles):

            optimizeur.zero_grad()

            lot_Y_predictions, etat = modele(lot_X, etat)
            cout = fonction_cout(lot_Y_predictions.transpose(1, 2), lot_Y)
            
            etat = etat.detach()

            cout.backward()
            optimizeur.step()

            print({ 'epoch': epoch, 'lot': lot, 'loss': cout.item() })

def predire(ds, modele, debut_texte, nb_mots=100):
    """ Prédire une suite de nb_mots à partir de debut_texte selon le modele"""
    mots = debut_texte.split(' ')
    modele.eval()
    etat = modele.initializer_etat(len(mots))

    for i in range(0, nb_mots):
        lot_X = torch.tensor([[ds.mot_a_index[m] for m in mots[i:]]])
        lot_Y_predictions, etat = modele(lot_X, etat)
        dernier_mot_Yt = lot_Y_predictions[0][-1]
        probs_dernier_mot = torch.nn.functional.softmax(dernier_mot_Yt, dim=0).detach().numpy()
        index_mot_choisi = np.random.choice(len(dernier_mot_Yt), p=probs_dernier_mot)
        mots.append(ds.index_a_mot[index_mot_choisi])

    return mots

ds_paroles = DatasetParoles(taille_sequence=6)
modele = Modele(ds_paroles)

entrainer_RNR(ds_paroles, modele, taille_lot=32, epochs=10, taille_sequence=6)
print(predire(ds_paroles, modele, debut_texte='I could'))
