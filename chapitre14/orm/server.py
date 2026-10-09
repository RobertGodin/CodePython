from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///exemple.db'
db = SQLAlchemy(app, model_class=Base)

# Définition du modèle (équivalent de la table users)
class User(db.Model):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    email: Mapped[str]

# Création de la base et de la table au premier lancement
with app.app_context():
    db.create_all()

@app.route('/')
def maison():
    # Récupère tous les utilisateurs
    users = db.session.execute(db.select(User)).scalars().all()
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
    # debug=True : seulement en développement, jamais en production
    app.run(debug=True)
