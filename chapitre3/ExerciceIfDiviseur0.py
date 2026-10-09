"""
Exercice if division. Si le diviseur est nul, afficher un message
"""

dividende = int(input("entrez un nombre entier, le dividende : "))
diviseur = int(input("entrez un nombre entier, le diviseur : "))
if diviseur != 0:
    quotient = dividende // diviseur
    print(f"{dividende} divisé par {diviseur} donne {quotient}")
else:
    print("Le diviseur ne peut être nul (0)")
