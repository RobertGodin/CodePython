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


trouve = False

print(enumerate(une_chaine))

for indice, un_caractere in enumerate(une_chaine) :
    if un_caractere == 'a' :
        trouve = True
        indice_a = indice
        break
if trouve :
    print("Le caractère a est présent à la position :", indice_a)
else :
    print("Le caractère a n'est pas dans la chaîne")
