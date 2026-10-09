import sqlite3
from contextlib import closing
from datetime import datetime

import exifread
from flask import Flask, render_template, request, redirect

def get_if_exist(data, key):
    if key in data:
        return data[key]
    return None

def convert_to_degrees(value):
    """
    Convertit les coordonnées GPS des données EXIF en degrés décimaux
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)

def get_exif_location(exif_data):
    """
    Renvoie la latitude et la longitude, si elles sont disponibles,
    à partir des données exif_data (obtenues avec get_exif_data)
    """
    lat = None
    lon = None

    gps_latitude = get_if_exist(exif_data, 'GPS GPSLatitude')
    gps_latitude_ref = get_if_exist(exif_data, 'GPS GPSLatitudeRef')
    gps_longitude = get_if_exist(exif_data, 'GPS GPSLongitude')
    gps_longitude_ref = get_if_exist(exif_data, 'GPS GPSLongitudeRef')

    if (gps_latitude and gps_latitude_ref
            and gps_longitude and gps_longitude_ref):
        lat = convert_to_degrees(gps_latitude)
        if gps_latitude_ref.values[0] != 'N':
            lat = 0 - lat

        lon = convert_to_degrees(gps_longitude)
        if gps_longitude_ref.values[0] != 'E':
            lon = 0 - lon

    return lat, lon

def get_exif_data(image_file):
    with open(image_file, 'rb') as f:
        exif_tags = exifread.process_file(f)
    return exif_tags

def log(long, lat):
    # closing() ferme la connexion à la fin du bloc
    with closing(sqlite3.connect("img.db")) as con:
        # with con valide la transaction (commit) à la fin du bloc
        with con:
            tables = [row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            if "geo" not in tables:
                con.execute("CREATE TABLE geo"
                            " (date TEXT, long NUMERIC, lat NUMERIC)")
            dt = datetime.now()
            con.execute("INSERT INTO geo (date, long, lat)"
                        " VALUES (?, ?, ?)", (str(dt), long, lat))

app = Flask(__name__)
# Refuser les fichiers de plus de 16 Mo (erreur 413)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return redirect('/upload')

@app.route('/upload')
def upload_file_render():
    return render_template('formulaire.html',
                           message="Choisissez une image JPEG.")

@app.route('/uploader', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if f is None or not f.filename.lower().endswith(('.jpg', '.jpeg')):
        return render_template('formulaire.html',
                               message="Choisissez une image JPEG.")
    lat, long = get_exif_location(exifread.process_file(f))
    if lat is None:
        return render_template('formulaire.html',
                               message="Il n'y avait pas de données GPS"
                               " dans l'image. Choisissez une autre image.")
    link = ("https://www.openstreetmap.org/"
            f"?mlat={lat}&mlon={long}&zoom=15")
    log(long, lat)
    return f'<html><body><a href="{link}">carte</a></body></html>'

if __name__ == '__main__':
    app.run()
