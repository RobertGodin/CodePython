# Importer le module sanic, qui est un framework web
import sanic

# Importer le module random, pour les nombres aléatoires
import random

# Importer le module asyncio, qui nous permet de faire des pauses
import asyncio

# Créer une instance de l'application sanic
app = sanic.Sanic(__name__)

# Définir une liste de couleurs possibles
couleurs = ["red", "green", "blue", "yellow", "pink", "orange"]

# Définir une fonction asynchrone
async def envoyer_couleur(ws):
    # Choisir une couleur aléatoire dans la liste
    couleur = random.choice(couleurs)
    # Envoyer la couleur au client
    await ws.send(couleur)

# Définir une route qui renvoie une page HTML
@app.route("/")
async def index(request):
    return sanic.response.html("""<!DOCTYPE html>
    <html lang="fr">
        <head>
            <meta charset="utf-8">
            <title>Changer la couleur de la page</title>
        </head>
        <body>
            <h1>Changer la couleur de la page</h1>
            <script>
                // wss:// si la page est servie en HTTPS, sinon ws://
                const protocole =
                    location.protocol === "https:" ? "wss://" : "ws://";
                // Créer un objet websocket qui se connecte
                const ws = new WebSocket(protocole + location.host + "/ws");
                // Définir une fonction qui change la couleur
                const changer_couleur = function(couleur) {
                    document.body.style.backgroundColor = couleur;
                }
                // Quand le websocket reçoit un message du serveur
                ws.onmessage = function(event) {
                    // Récupérer la couleur envoyée par le serveur
                    const couleur = event.data;
                    // Appeler la fonction qui change la couleur de la page
                    changer_couleur(couleur);
                }
            </script>
        </body>
    </html>
    """)

# Définir une route websocket qui communique avec le client
@app.websocket("/ws")
async def websocket(request, ws):
    # Tant que le websocket est ouvert
    while True:
        # Appeler la fonction qui envoie une couleur
        await envoyer_couleur(ws)
        # Attendre une seconde
        await asyncio.sleep(1)

# Lancer l'application sanic
if __name__ == "__main__":
    app.run(host="localhost", port=8000)
