# -*- coding: utf-8 -*-
"""
@author: Robert
Exercice deux while imbriqués
"""
compteur1 = 1
while compteur1 <= 9:
    compteur2 = 1
    while compteur2  <= compteur1:
        print(compteur2, end='')
        compteur2 = compteur2  +1
    print('')
    compteur1 = compteur1  +1