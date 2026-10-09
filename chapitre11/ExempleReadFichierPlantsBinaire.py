"""
Lecture du fichier binaire Plants.dat encodé avec struct
"""

import struct

with open('Plants.dat', mode='rb') as fichier_binaire:
    while True:
        r = fichier_binaire.read(28)
        if not r:
            break
        print(struct.unpack('<i20sf', r))
