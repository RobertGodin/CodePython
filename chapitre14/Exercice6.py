# Importer le module datetime
from datetime import datetime

# Importer le module locale
import locale

# Importer le module flask
from flask import Flask

# Définir le paramètre régional une seule fois, au démarrage
try:
    locale.setlocale(locale.LC_TIME, "fr_CA.UTF-8")
except locale.Error:
    # Paramètre régional non installé : on garde celui par défaut
    pass

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir la route principale
@app.route("/")
def index():
    # Obtenir la date et l'heure
    now = datetime.now()

    # Formater la date et l'heure
    date = now.strftime("%A %d %B %Y")
    heure = now.strftime("%H:%M:%S")

    # Retourner une page web avec la date et l'heure
    return f"""
    <html>
        <head>
            <title>Date et heure</title>
        </head>
        <body>
            <h1>Bonjour, voici la date et l'heure:</h1>
            <p>Date: {date}</p>
            <p>Heure: {heure}</p>
        </body>
    </html>
    """

# Lancer l'application Flask
if __name__ == "__main__":
    app.run()
