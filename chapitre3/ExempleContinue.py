"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme
des entiers lus. Omettre les entiers négatifs. Exemple du continue.
"""

somme=0
while True:
    entier_lu = int(input("Entrez un nombre entier, 0 pour terminer :"))
    if entier_lu > 0 :
        somme = somme + entier_lu
    elif entier_lu < 0:
        continue
    else:
        break
print("La somme des entiers non négatifs est:",somme)
