from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///exemple.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Définition du modèle (équivalent de la table users)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, nullable=False)

# Création de la base et de la table au premier lancement
with app.app_context():
    db.create_all()

@app.route('/')
def maison():
    users = User.query.all()          # Récupère tous les utilisateurs
    return render_template('maison.html', users=users)

@app.route('/add', methods=['POST'])
def ajoute():
    name = request.form['name']
    email = request.form['email']
    
    new_user = User(name=name, email=email)  # Crée l'objet
    db.session.add(new_user)                 # Ajoute en attente
    db.session.commit()                      # Enregistre dans la base
    
    return redirect(url_for('maison'))

if __name__ == '__main__':
    app.run(debug=True)