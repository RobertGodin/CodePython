# -*- coding: utf-8 -*-
"""
Lire une chaîne et vérifier si 'a' est dans la chaîne avec un for
"""

une_chaine = input("Entrez une chaîne de caractères :")
trouve = False
for un_caractere in une_chaine :
    if un_caractere == 'a' :
        trouve = True
        break
print(trouve)