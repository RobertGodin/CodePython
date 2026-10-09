''' Exercice2.py
Ce programme saisit trois entiers et en affiche la somme en employant une seule
variable de type str et une de type int'''

chaine = input("Entrez un premier nombre entier :")
somme = int(chaine)

chaine = input("Entrez un second nombre entier :")
somme = somme + int(chaine)

chaine = input("Entrez un troisième nombre entier :")
somme = somme + int(chaine)

# Afficher la somme 
print("La somme des trois entiers est ",somme)
