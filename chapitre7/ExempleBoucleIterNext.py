# -*- coding: utf-8 -*-
une_liste = [3,1,2]
un_iterateur = iter(une_liste)
while True:
    try:
        suivant = next(un_iterateur)
        print(suivant)
    except StopIteration:
        print("Fin de l'itérable")
        break

for suivant in une_liste :
    print(suivant)