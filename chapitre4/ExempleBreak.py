"""
Lire une chaîne et vérifier si 'a' est dans la chaîne
"""

une_chaine = input("Entrez une chaîne de caractères :")
indice = 0
trouve = False
while indice < len(une_chaine) :
    if une_chaine[indice] == 'a' :
        trouve = True
        break
    indice += 1
print(trouve)
