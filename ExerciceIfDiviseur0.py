# -*- coding: utf-8 -*-
"""
Exercice if division. Si le diviseur est nul, afficher un message
"""

dividende = int(input("entrez un nombre entier, le dividende : "))
diviseur = int(input("entrez un nombre entier, le diviseur : "))
if (diviseur != 0):
    print("Le résultat de ", dividende, "divisé par",diviseur,"est :", dividende//diviseur)
else :
    print("Le diviseur ne peut être nul (0)")