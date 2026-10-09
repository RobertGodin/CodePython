import sqlite3
from contextlib import closing

from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/')
def maison():
    conn = sqlite3.connect('exemple.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    conn.close()
    return render_template('maison.html', users=users)

@app.route('/add', methods=['POST'])
def ajoute():
    name = request.form['name']
    email = request.form['email']

    conn = sqlite3.connect('exemple.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (name, email) VALUES (?, ?)',
              (name, email))
    conn.commit()
    conn.close()
    return redirect('/')

with closing(sqlite3.connect('exemple.db')) as conn:
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL);''')

if __name__ == '__main__':
    app.run()
