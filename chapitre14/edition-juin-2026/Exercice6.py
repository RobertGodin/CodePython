# Importer le module flask
from flask import Flask

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir la route principale
@app.route("/")
def index():
    # Importer le module datetime
    from datetime import datetime

    # Importer le module locale, qui permet de gérer les paramètres régionaux
    import locale

    # Définir le paramètre régional à "fr_CA" pour le français du Canada
    locale.setlocale(locale.LC_TIME, "fr_CA")

    # Obtenir la date et l'heure
    now = datetime.now()

    # Formater la date et l'heure en français avec la méthode strftime
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
