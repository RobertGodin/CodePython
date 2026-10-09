"""
Lecture du fichier Plants.csv en format csv
"""
import csv
with open('Plants.csv', newline='', encoding='utf-8') as fichier_csv:
    lecteur_csv = csv.reader(fichier_csv)
    for ligne in lecteur_csv:
        print(ligne)
