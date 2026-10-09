"""
Exemple de fonction surface qui retourne une valeur
Exception si un argument est négatif
"""
def surface(largeur, hauteur):
    if largeur < 0 or hauteur < 0 :
        raise ValueError("La largeur et la hauteur doivent être positives")
    return largeur*hauteur

if __name__ == "__main__":
    s = surface(3,5)
    print(f"La surface est : {s}")
