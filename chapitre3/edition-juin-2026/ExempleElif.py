# -*- coding: utf-8 -*-
"""
Afficher la note littérale correspondant à la note numérique avec la clause elif
"""

note = int(input("entrez une note entre 0 et 100 : "))
if note < 0:
    print("La note doit ne peut être inférieure à 0")
elif note < 60:
    print("E")
elif note < 70:
    print("D")
elif note < 80:
    print("C")
elif note < 90:
    print("B")
elif note <= 100:
    print("A")
else:
    print("La note ne peut être supérieure à 100")
    