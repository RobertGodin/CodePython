"""
Exemple du read() binaire
"""
with open("Fichier1.txt", "rb") as fichier:
    contenu_fichier = fichier.read()
print(f"Taille fichier: {len(contenu_fichier)}")
print("Contenu type bytes:")
print(contenu_fichier)
print("Contenu en binaire:")
for un_octet in contenu_fichier :
    print(f"{un_octet:08b}")
