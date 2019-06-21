# -*- coding: utf-8 -*-
"""
Exemple du read() entier en binaire
"""
fichier = open("Fichier1629696561.dat", "rb")
octets = fichier.read(4)
i = int.from_bytes(octets, byteorder='big', signed=True)
print(i)
fichier.close()
