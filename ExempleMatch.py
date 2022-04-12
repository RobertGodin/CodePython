# -*- coding: utf-8 -*-
"""
Afficher la note littérale correspondant à la note numérique avec match (python 3.10)
"""

note = int(input("entrez une note entre 0 et 100 : "))
match note :
    case note if (note < 0):
        print("La note doit ne peut être inférieure à 0")
    case note if (note < 60):
        print("E")
    case note if  (note < 70):
        print("D")
    case note if  (note < 80):
        print("C")
    case note if  (note < 90):
        print("B")
    case note if  (note <= 100):
        print("A")
    case _:
        print("La note ne peut être supérieure à 100")