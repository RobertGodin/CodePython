from flask import Flask
app = Flask(__name__)
@app.route('/utilisateur/<nom>')
def show_user_profile(nom):
    # Affiche le profil de l'utilisateur spécifié par <nom>
    return '<html><body>Bonjour %s</body></html>' % nom

if __name__ == '__main__':
    app.run()