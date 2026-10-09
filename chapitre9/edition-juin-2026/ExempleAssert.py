# -*- coding: utf-8 -*-
'''
Ce programme saisit deux entiers et en affiche le quotient. Exemple de assert. 
'''

chaine1 = input("Entrez le dividende :")
chaine2 = input("Entrez le diviseur :")

dividende = int(chaine1)
diviseur = int(chaine2)

assert diviseur != 0, "Le diviseur ne peut être 0"

quotient = dividende / diviseur
print("Le quotient est ",quotient)