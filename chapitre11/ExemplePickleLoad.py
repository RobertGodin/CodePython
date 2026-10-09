"""
Lecture du fichier Plants.pickle avec pickle.load()
"""

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

with open('Plants.pickle', mode = 'rb') as fichier_binaire:
    liste_plants = pickle.load(fichier_binaire)

for un_plant in liste_plants:
    print(un_plant)
