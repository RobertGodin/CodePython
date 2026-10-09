"""
Exemple de fonction surface qui retourne une valeur
"""
def surface(largeur, hauteur):
    return largeur*hauteur

if __name__ == "__main__":
    s = surface(3,5)
    print(f"La surface est : {s}")
