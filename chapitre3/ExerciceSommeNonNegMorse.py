"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme
des entiers lus. Omettre les entiers négatifs. Exemple avec opérateur morse.
"""

somme=0
while (entier_lu := int(input("Entrez un nombre entier, 0 pour terminer :"))) != 0:
    if entier_lu > 0 :
        somme = somme + entier_lu
print("La somme des entiers non négatifs est:",somme)
