# -*- coding: utf-8 -*-
"""
Lecture du fichier Plants.csv en format csv, création d'une liste d'objets de la classe Plant
Ecriture de l'objet liste_plants dans Plants.pickle avec le module pickle
"""

import csv
import struct
import json

# Classe Plant
class Plant:
    def __init__(self,numero,description,prix):
        self.numero = numero
        self.description = description
        self.prix = prix
    def __str__(self):
        return '|{:>10}|{:>20}|{:8.2f}|'.format(self.numero,self.description,self.prix)

liste_plants = []

with open('Plants.csv', mode='r') as fichier_csv:
    lecteur_csv = csv.reader(fichier_csv)
    for ligne in lecteur_csv:
        un_plant = Plant(ligne[0],ligne[1],ligne[2])
#        print(un_plant)
        liste_plants.append(ligne)

with open('Plants.json', mode = 'w') as fichier_json:
    json.dump(liste_plants, fichier_json)