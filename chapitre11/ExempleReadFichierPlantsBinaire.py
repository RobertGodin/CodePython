# -*- coding: utf-8 -*-
"""
Lecture du fichier binaire Plants.dat encodé avec struct
"""

import csv
import struct

with open('Plants.dat', mode='rb') as fichier_binaire:
    while True:
        enregistrement=fichier_binaire.read(28)
        print(enregistrement)
        if enregistrement == b'': #fin de fichier ?
            break
        print(struct.unpack('i20sf',enregistrement))