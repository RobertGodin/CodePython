# -*- coding: utf-8 -*-
"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme
des entiers lus. Exemple du try.
"""

somme=0
while True:
    chaine = input("Entrez un nombre entier, 0 pour terminer :")
    if chaine != '0' :
        try :
            somme = somme + int(chaine)
        except ValueError :
            print("La chaîne '" + chaine + "' n'est pas un nombre et sera exclue")
    else:
        break
print("La somme des entiers est:",somme)