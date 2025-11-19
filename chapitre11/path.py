from pathlib import Path

# chemin courant, dossier personnel, joindre des parties
p = Path("documents") / "python" / "exo.py"   # utilise automatiquement / ou \
print(p)                                      # → documents/python/exo.py

# lire et écrire du texte facilement
contenu = Path("path.py").read_text(encoding="utf-8")
Path("nouveau.txt").write_text("Bonjour le monde !", encoding="utf-8")

# lister, filtrer, créer des dossiers
for fichier in Path(".").glob("*.py"):        # tous les .py du dossier courant
    print(fichier.name)

Path("dossier/temp/niveau3").mkdir(parents=True, exist_ok=True)
