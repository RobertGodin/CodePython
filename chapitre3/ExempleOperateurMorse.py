"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme des entiers lus.
Solution avec l'opérateur morse :=
"""

somme = 0
invite = "Entrez un nombre entier, 0 pour terminer :"
while (chaine := input(invite)) != '0':
    somme += int(chaine)
print(f"La somme des entiers est : {somme}")
