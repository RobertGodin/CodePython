# -*- coding: utf-8 -*-
"""
Exemples d'affichage du contenu d'un répertoire avec os.listdir() et os.walk()
"""

import os

repertoire_de_travail = os.getcwd() # Chercher le répertoire de travail courant
print("Répertoire de travail :",repertoire_de_travail)

print("Éléments de permier niveau du répertoire de travail :")
print(os.listdir(repertoire_de_travail))

print("Parcourir le répertoire de travail niveau par niveau :")
i=0
for chemin, dirnames, filenames in os.walk(repertoire_de_travail):
    i+=1
    if i > 50 :
        break
    print("------------- Chemin :", chemin)
    print("Noms des répertoire:",dirnames)
    print("Noms des fichiers:",filenames[:3])
