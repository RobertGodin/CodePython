# -*- coding: utf-8 -*-
"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme des entiers lus.
"""

somme=0
chaine = input("Entrez un nombre entier, 0 pour terminer :")
while chaine != '0':
    somme = somme + int(chaine)
    chaine = input("Entrez un nombre entier, 0 pour terminer :")
print("La somme des entiers est:",somme)