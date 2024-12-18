# -*- coding: utf-8 -*-
"""
Lecture du fichier Plants.pickle avec pickle.load()
"""

import csv
import struct
import pickle

# Classe Plant
class Plant:
    def __init__(self,numero,description,prix):
        self.numero = numero
        self.description = description
        self.prix = prix
    def __str__(self):
        return '|{:>10}|{:>20}|{:8.2f}|'.format(self.numero,self.description,self.prix)

with open('Plants.pickle', mode = 'rb') as fichier_binaire: 
    liste_plants = pickle.load(fichier_binaire)

for un_plant in liste_plants:
    print(un_plant)
