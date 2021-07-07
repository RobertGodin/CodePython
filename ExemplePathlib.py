# -*- coding: utf-8 -*-
"""
Exemple de chemin avec Pathlib compatible Windows ou Unix
"""

from pathlib import Path
import os

repertoire = Path("/Users/Robert/CodePython/data")

os.chdir(repertoire) # Changer le répertoire de travail

nouveau_repertoire_de_travail = os.getcwd() # Chercher le nouveau répertoire de travail
print("Nouveau répertoire de travail :",nouveau_repertoire_de_travail)

print("Contenu du répertoire de travail:")
print(os.listdir(nouveau_repertoire_de_travail))

