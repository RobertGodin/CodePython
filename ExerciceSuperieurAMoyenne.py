# -*- coding: utf-8 -*-
"""
Lire 5 entiers, afficher la moyenne et les entiers supérieurs à la moyenne.
"""
somme = 0
liste_de_int = []
for indice in range(5):
    un_int = int(input('Entrez un entier:'))
    somme = somme + un_int
    liste_de_int = liste_de_int + [un_int]
moyenne = somme / 5
print('La moyenne est :',moyenne)
print('Liste des entiers lus supérieurs à la moyenne :')
for indice in range(5):
    if liste_de_int[indice] > moyenne :
        print(liste_de_int[indice])