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
        return (f"|{self.numero:>10}|{self.description:>20}"
                f"|{self.prix:8.2f}|")

with open('Plants.json', mode='r', encoding='utf-8') as fichier_json:
    donnees = json.load(fichier_json)

# Reconstruire les objets Plant à partir des dictionnaires
liste_plants = [Plant(**d) for d in donnees]

for un_plant in liste_plants:
    print(un_plant)
