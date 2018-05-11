# -*- coding: utf-8 -*-
"""
Exemple afficher le maximum de trois entiers
"""

entier1 = int(input("entrez un nombre entier: "))
entier2 = int(input("entrez un nombre entier: "))
entier3 = int(input("entrez un nombre entier: "))

if (entier1 > entier2):
    if (entier1 > entier3):
        print("Le maximum des trois entiers est :", entier1)
    else:
        print("Le maximum des trois entiers est :", entier3)
else:
    if (entier2 > entier3):
        print("Le maximum des trois entiers est :", entier2)
    else:
        print("Le maximum des trois entiers est :", entier3)
