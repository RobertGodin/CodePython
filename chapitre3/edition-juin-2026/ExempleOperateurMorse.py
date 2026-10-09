# -*- coding: utf-8 -*-
"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme des entiers lus.
Solution avec l'opérateur morse :=
"""

somme=0 
while (chaine := input("Entrez un nombre entier, 0 pour terminer :")) != '0':
    somme = somme + int(chaine)
print("La somme des entiers est:", somme)

