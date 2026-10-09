import json

import simdjson
from roaringbitmap import RoaringBitmap

# Créer un objet RoaringBitmap
rb = RoaringBitmap()

# Ajouter des éléments à l'objet RoaringBitmap
rb.add(1)
rb.add(2)
rb.add(3)

# Convertir le contenu de l'objet RoaringBitmap en JSON
json_data = json.dumps(list(rb))

# Analyser le JSON avec simdjson (paquet pysimdjson)
data = simdjson.loads(json_data)

# Afficher les éléments de l'objet RoaringBitmap
for i in data:
    print(i)
