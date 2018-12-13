# -*- coding: utf-8 -*-
"""
Exemple de lecture d'un fichier en mode texte
"""

fichier = open("Fichier1.txt", "r")
for ligne in fichier:
    print(ligne,end="")
fichier.close()