# -*- coding: utf-8 -*-
"""
Exemple de fonction surface qui retourne une valeur
Exception si un paramètre est négatif
"""
def surface(largeur, hauteur):
    if largeur < 0 or hauteur < 0 :
        raise(Exception)
    return largeur*hauteur
s = surface(3,5)
print("La surface est :",s)
