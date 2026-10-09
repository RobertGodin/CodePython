"""
Lecture du fichier Plants.csv en format csv et stockage en un fichier binaire
Conversion des données avec struct
"""

import csv
import struct

with open('Plants.csv', newline='', encoding='utf-8') as fichier_csv:
    with open('Plants.dat', mode='wb') as fichier_binaire:
        lecteur_csv = csv.reader(fichier_csv)
        for ligne in lecteur_csv:
            octets = struct.pack('<i20sf', int(ligne[0]),
                                 ligne[1].encode('utf-8'),
                                 float(ligne[2]))
            fichier_binaire.write(octets)
