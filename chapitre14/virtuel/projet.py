import simdjson
from roaringbitmap import RoaringBitmap

# Créer un objet RoaringBitmap
rb = RoaringBitmap()

# Ajouter des éléments à l'objet RoaringBitmap
rb.add(1)
rb.add(2)
rb.add(3)

# Convertir l'objet RoaringBitmap en JSON
json_data = str([i for i in rb])

# Analyser le JSON avec pysimdjson
data = simdjson.loads(json_data)

# Afficher les éléments de l'objet RoaringBitmap
for i in data:
    print(i)
