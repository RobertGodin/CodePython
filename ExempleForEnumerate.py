# -*- coding: utf-8 -*-
"""
Lire une chaîne et vérifier si 'a' est dans la chaîne. 
Afficher la position en passant par un for avec enumerate.
"""

une_chaine = input("Entrez une chaîne de caractères :")
trouve = False

for indice, un_caractere in enumerate(une_chaine) :
    if un_caractere == 'a' :
        trouve = True
        indice_a = indice
        break
if trouve :
    print("Le caractère a est présent à la position :", indice_a)
else :
    print("Le caractère a n'est pas dans la chaîne")
