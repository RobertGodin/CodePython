"""
Exemple du write() entier en binaire
"""
i = 1629696561
with open("Fichier1629696561.dat", "wb") as fichier:
    fichier.write(i.to_bytes(4, byteorder='big', signed=True))
