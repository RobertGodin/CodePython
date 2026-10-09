"""
Exemple de lecture d'un fichier en mode texte
"""
fichier = open("Fichier1.txt", "rt", encoding="utf-8")
for ligne in fichier:
    print(ligne,end="")
fichier.close()
