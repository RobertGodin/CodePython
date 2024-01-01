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
for un_int in liste_de_int:
    if un_int> moyenne :
        print(un_int)