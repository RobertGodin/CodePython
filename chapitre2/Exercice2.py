''' Exercice2.py
Saisit trois entiers et en affiche la somme avec une seule
variable de type str et une de type int'''

chaine = input("Entrez un premier nombre entier :")
somme = int(chaine)

chaine = input("Entrez un second nombre entier :")
somme += int(chaine)

chaine = input("Entrez un troisième nombre entier :")
somme += int(chaine)

# Afficher la somme
print(f"La somme des trois entiers est {somme}")
