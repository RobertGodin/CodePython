"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme
des entiers lus.Exemple du break.
"""

somme=0
while True:
    chaine = input("Entrez un nombre entier, 0 pour terminer :")
    if chaine != '0' :
        somme = somme + int(chaine)
    else:
        break
print("La somme des entiers est:",somme)
