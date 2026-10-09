# -*- coding: utf-8 -*-
"""
Exemple de modification d'une liste par une fonction
"""

def f2(une_liste):
    une_liste[0] = 1
liste = [5,6,7] 
print("Valeur de la liste avant l'appel de f2:",liste)
f2(liste)
print("Valeur de la liste après l'appel de f2:",liste)