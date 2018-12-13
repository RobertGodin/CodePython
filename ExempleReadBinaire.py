# -*- coding: utf-8 -*-
"""
Exemple du read() binaire
"""
fichier = open("Plants.csv", "rb")
contenu_fichier = fichier.read()
print("Taille fichier:", len(contenu_fichier))
print("Contenu type bytes:")
print(contenu_fichier)
print("Contenu en binaire:")
for un_octet in contenu_fichier :
    print("{:0=8b}".format(un_octet))
fichier.close()

