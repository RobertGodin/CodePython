"""
Exemple du with
"""
with open("Fichier1.txt", "r", encoding="utf-8") as fichier:
    for ligne in fichier:
        print(ligne,end="")
