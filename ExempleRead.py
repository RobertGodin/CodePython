# -*- coding: utf-8 -*-
"""
Exemple du read()
"""
fichier = open("Fichier1.txt", "r")
contenu_fichier = fichier.read()
print("Taille fichier:", len(contenu_fichier))
print("Contenu:")
print(contenu_fichier)
fichier.close()
