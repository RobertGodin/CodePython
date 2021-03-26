# -*- coding: utf-8 -*-
"""
Exemple afficher le maximum de 5 entiers
"""

le_maximum_actuel = int(input("entrez un nombre entier: "))
for compteur in range(5):
    entier_lu = int(input("entrez un nombre entier: "))
    if (entier_lu > le_maximum_actuel):
        le_maximum_actuel = entier_lu
print("Le maximum des 5 entiers lus est:",le_maximum_actuel3
      )    
