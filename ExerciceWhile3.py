# -*- coding: utf-8 -*-
"""
@author: Robert
Lire dix entiers et en afficher la somme avec un while
"""

somme=0
compteur = 1
while(compteur <= 10):
    chaine = input("Entrez un nombre entier:")
    somme = somme + int(chaine)
    compteur = compteur+1
print("La somme des dix entiers est:",somme)
