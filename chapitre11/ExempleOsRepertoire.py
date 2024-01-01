# -*- coding: utf-8 -*-
"""
Exemples de base de manipulation de répertoire de fichier avec le module os
"""

import os

repertoire_de_travail = os.getcwd() # Chercher le répertoire de travail
print("Répertoire de travail :",repertoire_de_travail)

# Création d'un chemin par concaténation avec os.path.join
sous_repertoire = os.path.join(repertoire_de_travail, "repertoire_test")

if not os.path.exists(sous_repertoire):
    print("Création du sous-répertoire 'repertoire_test' dans le répertoire de travail")
    os.mkdir(sous_repertoire)

print("Changement du répertoire de travail")
os.chdir(sous_repertoire) # Changer le répertoire de travail

nouveau_repertoire_de_travail = os.getcwd() # Chercher le nouveau répertoire de travail
print("Nouveau répertoire de travail :",nouveau_repertoire_de_travail)

print("Création du fichier 'FichierTest.txt' dans le répertoire de travail")
fichier = open("FichierTest.txt", "w")
fichier.write("abc\n12\n")
fichier.close()

print("Contenu du répertoire de travail:")
print(os.listdir(nouveau_repertoire_de_travail))
