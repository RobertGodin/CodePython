from flask import Flask
from markupsafe import escape

app = Flask(__name__)

@app.route('/utilisateur/<nom>')
def show_user_profile(nom):
    # Affiche le profil de l'utilisateur spécifié par <nom>
    # escape() neutralise le HTML qu'un visiteur glisserait dans l'URL
    return f'<html><body>Bonjour {escape(nom)}</body></html>'

if __name__ == '__main__':
    app.run()
