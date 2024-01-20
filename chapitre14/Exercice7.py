# Importer le module flask
from flask import Flask

# Importer le module sqlite3, qui permet de manipuler des bases de données sqlite
import sqlite3

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir le nom du fichier de la base de données
db_file = "access.db"

# Définir la fonction qui crée la table de la base de données si elle n'existe pas
def create_table():
    # Se connecter à la base de données
    conn = sqlite3.connect(db_file)
    # Créer un curseur pour exécuter des commandes SQL
    cursor = conn.cursor()
    # Créer la table access si elle n'existe pas, avec deux colonnes: id et date
    cursor.execute("CREATE TABLE IF NOT EXISTS access (id INTEGER PRIMARY KEY, date TEXT)")
    # Valider les changements
    conn.commit()
    # Fermer la connexion
    conn.close()

# Définir la fonction qui insère la date du dernier accès dans la base de données
def insert_date(date):
    # Se connecter à la base de données
    conn = sqlite3.connect(db_file)
    # Créer un curseur pour exécuter des commandes SQL
    cursor = conn.cursor()
    # Insérer la date dans la table access, en laissant l'id se générer automatiquement
    cursor.execute("INSERT INTO access (date) VALUES (?)", (date,))
    # Valider les changements
    conn.commit()
    # Fermer la connexion
    conn.close()

# Définir la fonction qui récupère le contenu de la table access
def get_table_content():
    # Se connecter à la base de données
    conn = sqlite3.connect(db_file)
    # Créer un curseur pour exécuter des commandes SQL
    cursor = conn.cursor()
    # Sélectionner toutes les lignes de la table access
    cursor.execute("SELECT * FROM access")
    # Récupérer le résultat sous forme de liste de tuples
    result = cursor.fetchall()
    # Fermer la connexion
    conn.close()
    # Retourner le résultat
    return result

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

    # Créer la table de la base de données si elle n'existe pas
    create_table()

    # Insérer la date du dernier accès dans la base de données
    insert_date(date)

    # Récupérer le contenu de la table access
    content = get_table_content()

    # Créer une variable pour stocker le code HTML du tableau
    table = ""

    # Parcourir le contenu de la table
    for row in content:
        # Ajouter une ligne au tableau avec les valeurs de l'id et de la date
        table += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"

    # Retourner une page web avec la date et l'heure et le tableau
    return f"""
    <html>
        <head>
            <title>Date et heure</title>
        </head>
        <body>
            <h1>Bonjour, voici la date et l'heure:</h1>
            <p>Date: {date}</p>
            <p>Heure: {heure}</p>
            <h2>Voici le contenu de la table access:</h2>
            <table border="1">
                <tr><th>Id</th><th>Date</th></tr>
                {table}
            </table>
        </body>
    </html>
    """

# Lancer l'application Flask
if __name__ == "__main__":
    app.run()

