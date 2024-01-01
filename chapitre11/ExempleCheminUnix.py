# -*- coding: utf-8 -*-
"""
Exemple de chemin selon la syntaxe Unix
"""

import os

os.chdir("/Users/Robert/CodePython/data") # Changer le répertoire de travail

nouveau_repertoire_de_travail = os.getcwd() # Chercher le nouveau répertoire de travail
print("Nouveau répertoire de travail :",nouveau_repertoire_de_travail)

print("Contenu du répertoire de travail:")
print(os.listdir(nouveau_repertoire_de_travail))