"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme des entiers lus.
"""

somme = 0
chaine = input("Entrez un nombre entier, 0 pour terminer :")
while chaine != '0':
    somme += int(chaine)
    chaine = input("Entrez un nombre entier, 0 pour terminer :")
print(f"La somme des entiers est : {somme}")
