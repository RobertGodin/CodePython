import exifread
from flask import Flask, render_template, request, redirect

def get_if_exist(data, key):
    if key in data:
        return data[key]
    return None


def convert_to_degrees(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)
    
def get_exif_location(exif_data):
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




app = Flask(__name__)

@app.route('/uploader', methods = ['POST'])
def upload_file():
    print("upload_file")
    f = request.files['file']
    lat, long = get_exif_location(exifread.process_file(f))
    print(lat, long)
    if lat is None:
        return render_template('formulaire.html', 
                               message = "Il n'y avait pas de données GPS dans l'image."
                                +" Choisissez une autre image.")
    link = "https://www.openstreetmap.org/?mlat="+str(lat)+"&mlon="+str(long)+"&zoom=15"
    return "<html><body><a href=\""+link+"\">carte</a></body></html>"

@app.route('/')
def index():
    return redirect('/upload')

@app.route('/upload')
def upload_file_render():
   print("upload_file_render")
   return render_template('formulaire.html', message = "Choisissez une image JPEG.")
	
app.run()