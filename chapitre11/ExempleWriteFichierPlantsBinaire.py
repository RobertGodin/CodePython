# -*- coding: utf-8 -*-
"""
Lecture du fichier Plants.csv en format csv et stockage en un fichier binaire
Conversion des données avec struct
"""

import csv
import struct

with open('Plants.csv', mode='r') as fichier_csv:
    with open('Plants.dat',mode = 'wb') as fichier_binaire:
        lecteur_csv = csv.reader(fichier_csv)
        for ligne in lecteur_csv:
            fichier_binaire.write(struct.pack('i20sf',int(ligne[0]),str.encode(ligne[1]),float(ligne[2])))
