"""
Exemple de chemin avec pathlib compatible Windows ou Unix
"""

from pathlib import Path

# Path.home() : C:\Users\Robert sous Windows, /Users/Robert sous macOS
repertoire = Path.home() / "CodePython" / "data"
print(f"Répertoire : {repertoire}")

print("Contenu du répertoire :")
# iterdir() parcourt le répertoire, sans avoir à faire os.chdir()
print([element.name for element in repertoire.iterdir()])
