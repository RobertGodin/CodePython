"""
Lecture du fichier Plants.csv en format csv, création d'une liste d'objets de la classe Plant
Ecriture de l'objet liste_plants dans Plants.pickle avec le module pickle
"""

import csv
import pickle

# Classe Plant
class Plant:
    def __init__(self,numero,description,prix):
        self.numero = numero
        self.description = description
        self.prix = prix
    def __str__(self):
        return (f"|{self.numero:>10}|{self.description:>20}"
                f"|{self.prix:8.2f}|")

liste_plants = []

with open('Plants.csv', newline='', encoding='utf-8') as fichier_csv:
    lecteur_csv = csv.reader(fichier_csv)
    for ligne in lecteur_csv:
        un_plant = Plant(int(ligne[0]),ligne[1],float(ligne[2]))
        print(un_plant)
        liste_plants.append(un_plant)

with open('Plants.pickle', mode = 'wb') as fichier_binaire:
    pickle.dump(liste_plants, fichier_binaire, pickle.HIGHEST_PROTOCOL)
