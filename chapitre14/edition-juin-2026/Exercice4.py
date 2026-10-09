# Importer le module json
import json

# Ouvrir le fichier JSON qui contient la liste de numéros de téléphone
with open("exercice4.json", "r") as f:
    # Charger les données JSON dans une variable
    data = json.load(f)

# Parcourir la liste de numéros de téléphone
for item in data:
    # Vérifier si le nom correspond à John Galt
    if item["name"] == "John Galt":
        # Afficher le numéro de téléphone de John Galt
        print("Le numéro de téléphone de John Galt est:", item["phone"])
        # Arrêter la boucle
        break

