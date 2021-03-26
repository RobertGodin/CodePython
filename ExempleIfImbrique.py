# -*- coding: utf-8 -*-
"""
Exemple if sans else et if imbriqué
"""

un_int = int(input("entrez un nombre entier: "))
if (un_int > 10):
    if (un_int > 20):
        print(un_int,"est plus grand que 20")
    else :
        print(un_int,"est plus grand que 10 et plus petit ou égal à 20")
