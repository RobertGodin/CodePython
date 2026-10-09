# Importer le module asyncio
import asyncio

# Fonction ordinaire (bloquante) qui lit un fichier texte
def lire(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        return fichier.read()

# Définir une fonction asynchrone qui lit un fichier et l'affiche
async def lire_fichier(nom):
    # Lire le fichier dans un fil d'exécution séparé (to_thread) :
    # la boucle d'événements reste libre pendant la lecture
    contenu = await asyncio.to_thread(lire, nom)
    # Afficher le nom et le contenu du fichier
    print(f"Le fichier {nom} contient:\n{contenu}")

# Définir une fonction principale qui lance les tâches asynchrones
async def main():
    # Créer deux tâches dans un groupe de tâches ; la sortie du bloc
    # async with attend que les deux tâches soient terminées
    async with asyncio.TaskGroup() as groupe:
        groupe.create_task(lire_fichier("fichier1.txt"))
        groupe.create_task(lire_fichier("fichier2.txt"))

# Exécuter la fonction principale
asyncio.run(main())
