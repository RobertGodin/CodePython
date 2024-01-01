# -*- coding: utf-8 -*-
"""
Exemple du with
"""
with open("Fichier1.txt", "r") as fichier:
    for ligne in fichier:
        print(ligne,end="")