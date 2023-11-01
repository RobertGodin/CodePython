import exifread
from flask import Flask, render_template, request, redirect
from datetime import datetime
import sqlite3

def get_if_exist(data, key):
    if key in data:
        return data[key]
    return None


def convert_to_degrees(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
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
    Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
    """
    lat = None
    lon = None

    gps_latitude = get_if_exist(exif_data, 'GPS GPSLatitude')
    gps_latitude_ref = get_if_exist(exif_data, 'GPS GPSLatitudeRef')
    gps_longitude = get_if_exist(exif_data, 'GPS GPSLongitude')
    gps_longitude_ref = get_if_exist(exif_data, 'GPS GPSLongitudeRef')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
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


def log(long,lat):
  with sqlite3.connect("img.db") as con:
    tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if not "geo" in tables:
        con.execute("CREATE TABLE geo (date TEXT, long NUMERIC, lat NUMERIC)")
    dt = datetime.now()
    con.execute("INSERT INTO geo (date,long,lat) values (\""+str(dt)+"\","+str(long)+", "+str(lat)+") ")


app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/upload')

@app.route('/upload')
def upload_file_render():
   return render_template('formulaire.html', message = "Choisissez une image JPEG.")
	
@app.route('/uploader', methods = ['POST'])
def upload_file():
    f = request.files['file']
    f = request.files['file']
    lat, long = get_exif_location(exifread.process_file(f))
    if lat is None:
        return render_template('formulaire.html', "Il n'y avait pas de données GPS dans l'image. Choisissez une autre image.")
    link = "https://www.openstreetmap.org/?mlat="+str(lat)+"&mlon="+str(long)+"&zoom=15"
    log(long,lat)
    return "<html><body><a href=\""+link+"\">carte</a></body></html>"
		
app.run()