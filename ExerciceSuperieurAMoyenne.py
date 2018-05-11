# -*- coding: utf-8 -*-
"""
Lire 5 entiers, afficher la moyenne et les entiers supérieurs à la moyenne.
"""
somme = 0
listeDeInt = []
for indice in range(5):
    unInt = int(input('Entrez un entier:'))
    somme = somme + unInt
    listeDeInt = listeDeInt + [unInt]
moyenne = somme / 5
print('La moyenne est :',moyenne)
print('Liste des entiers lus supérieurs à la moyenne :')
for indice in range(5):
    if listeDeInt[indice] > moyenne :
        print(listeDeInt[indice])