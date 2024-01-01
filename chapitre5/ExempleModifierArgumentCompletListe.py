# -*- coding: utf-8 -*-
"""
Exemple de modification de l'argument complet de type liste par une fonction
Même effet qu'un passage par valeur
"""

def f3(une_liste):
    une_liste = [1,2,3,4]
liste = [5,6,7]
print("Valeur de la liste avant l'appel de f3:",liste)
f3(liste)
print("Valeur de la liste après l'appel de f3:",liste)
