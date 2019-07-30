# -*- coding: utf-8 -*-
"""
Lecture du fichier Plants.json avec json.load()
"""

import json

# Classe Plant
class Plant:
    def __init__(self,numero,description,prix):
        self.numero = numero
        self.description = description
        self.prix = prix
    def __str__(self):
        return '|{:>10}|{:>20}|{:8.2f}|'.format(self.numero,self.description,self.prix)

with open('Plants.json', mode = 'r') as fichier_json: 
    liste_plants = json.load(fichier_json)

for un_plant in liste_plants:
    print(un_plant)
