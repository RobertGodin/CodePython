# -*- coding: utf-8 -*-
"""
Exemple du write() entier en binaire
"""
fichier = open("Fichier1629696561.dat", "wb")
i = 1629696561
fichier.write(i.to_bytes(4, byteorder='big', signed=True))
fichier.close()