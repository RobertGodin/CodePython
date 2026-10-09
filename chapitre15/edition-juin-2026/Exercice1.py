# Importer le module asyncio, qui permet de gérer la programmation asynchrone
import asyncio

# Définir une fonction asynchrone qui lit un fichier et affiche son contenu
async def lire_fichier(nom):
    # Ouvrir le fichier en mode lecture
    with open(nom, "r") as fichier:
        # Lire le contenu du fichier
        contenu = fichier.read()
        # Afficher le nom et le contenu du fichier
        print(f"Le fichier {nom} contient:\n{contenu}")

# Définir une fonction principale qui lance les tâches asynchrones
async def main():
    # Créer deux tâches asynchrones pour lire deux fichiers différents
    tache1 = asyncio.create_task(lire_fichier("fichier1.txt"))
    tache2 = asyncio.create_task(lire_fichier("fichier2.txt"))
    # Attendre que les deux tâches soient terminées
    await tache1
    await tache2

# Exécuter la fonction principale avec la boucle d'événements d'asyncio
asyncio.run(main())

