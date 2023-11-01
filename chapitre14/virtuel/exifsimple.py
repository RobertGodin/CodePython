import exifread
# Ouvrir le fichier image pour la lecture (mode binaire)
f = open('img.jpg', 'rb')

# Extraire les tags EXIF
tags = exifread.process_file(f)

# Parcourir tous les tags et afficher les valeurs
for tag in tags.keys():
    if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
        print("Key: %s, value %s" % (tag, tags[tag]))
