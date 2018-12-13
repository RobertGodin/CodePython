# -*- coding: utf-8 -*-
"""
Lecture du fichier Plants.csv en format csv
"""

import csv

with open('Plants.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for ligne in csv_reader:
        print(ligne)