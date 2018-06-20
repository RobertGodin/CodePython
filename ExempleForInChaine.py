# -*- coding: utf-8 -*-
"""
Lire une chaîne et vérifier si 'a' est dans la chaîne avec un for
"""

uneChaine = input("Entrez une chaîne de caractères :")
indice = 0
trouve = False
for unCaractere in uneChaine :
    if unCaractere == 'a' :
        trouve = True
        break
print(trouve)