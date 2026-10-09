# -*- coding: utf-8 -*-
"""
Lecture en accès direct avec seek() dans le fichier binaire Plants.dat encodé avec struct
"""

import csv
import struct

with open('Plants.dat', mode='rb') as fichier_binaire:
    fichier_binaire.seek(56)
    enregistrement=fichier_binaire.read(28)
    print(enregistrement)
    print(struct.unpack('i20sf',enregistrement))