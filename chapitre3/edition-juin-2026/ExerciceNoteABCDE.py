# -*- coding: utf-8 -*-
"""
Afficher la note littérale correspondant à la note numérique
"""

note = int(input("entrez une note entre 0 et 100 : "))
if note < 0:
    print("La note doit ne peut être inférieure à 0")
else:
    if note < 60:
        print("E")
    else :
        if note < 70:
            print("D")
        else:
            if note < 80:
                print("C")
            else:
                if note < 90:
                    print("B")
                else:
                    if note <= 100:
                        print("A")
                    else:
                        print("La note ne peut être supérieure à 100")
    