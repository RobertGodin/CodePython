"""
Afficher le maximum de 5 entiers lus
"""

le_maximum_actuel = int(input("entrez un nombre entier: "))
for _ in range(4):
    entier_lu = int(input("entrez un nombre entier: "))
    if entier_lu > le_maximum_actuel:
        le_maximum_actuel = entier_lu
print(f"Le maximum des 5 entiers lus est : {le_maximum_actuel}")
