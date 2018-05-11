# -*- coding: utf-8 -*-
"""
Exemple if sans else et if imbriqué
"""

unInt = int(input("entrez un nombre entier: "))
if (unInt > 10):
    if (unInt > 20):
        print(unInt,"est plus grand que 20")
    else :
        print(unInt,"est plus grand que 10 et plus petit ou égal à 20")
