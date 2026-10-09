"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme
des entiers lus. Omettre les entiers négatifs. Exemple avec opérateur morse.
"""

somme = 0
invite = "Entrez un nombre entier, 0 pour terminer :"
while (entier_lu := int(input(invite))) != 0:
    if entier_lu > 0:
        somme += entier_lu
print(f"La somme des entiers non négatifs est : {somme}")
