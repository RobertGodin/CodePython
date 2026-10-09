"""
Lecture du fichier Plants.csv en format csv, création d'une liste d'objets de la classe Plant
Ecriture des données de liste_plants dans Plants.json avec json.dump()
"""

import csv
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

liste_plants = []

with open('Plants.csv', newline='', encoding='utf-8') as fichier_csv:
    lecteur_csv = csv.reader(fichier_csv)
    for ligne in lecteur_csv:
        un_plant = Plant(int(ligne[0]),ligne[1],float(ligne[2]))
        liste_plants.append(un_plant)

# JSON ne connaît pas la classe Plant : on écrit un dictionnaire par plant
donnees = [vars(un_plant) for un_plant in liste_plants]
with open('Plants.json', mode='w', encoding='utf-8') as fichier_json:
    json.dump(donnees, fichier_json, ensure_ascii=False, indent=2)
