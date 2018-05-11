# -*- coding: utf-8 -*-
"""
Exemple afficher le maximum de 5 entiers
"""

leMaximumActuel = int(input("entrez un nombre entier: "))
for compteur in range(5):
    entierLu = int(input("entrez un nombre entier: "))
    if (entierLu > leMaximumActuel):
        leMaximumActuel = entierLu
print("Le maximum des 5 entiers lus est:",leMaximumActuel)    
