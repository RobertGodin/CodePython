# -*- coding: utf-8 -*-
"""
Lire une chaîne et compter vérifier si 'a' est dans la chaîne
"""

uneChaine = input("Entrez une chaîne de caractères :")
indice = 0
trouve = False
while indice < len(uneChaine) and not(trouve) :
    if uneChaine[indice] == 'a' :
        trouve = True
    indice = indice + 1
print(trouve)