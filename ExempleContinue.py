"""
Lire une suite d'entiers jusqu'à ce que l'entier 0 soit entré et afficher la somme
des entiers lus. Omettre les entiers négatifs. Exemple du continue.
"""

somme=0
while True:
    entierLu = int(input("Entrez un nombre entier, 0 pour terminer :"))
    if entierLu > 0 :
        somme = somme + entierLu
    elif entierLu < 0:
        continue
    else:
        break
print("La somme des entiers non négatifs est:",somme)
