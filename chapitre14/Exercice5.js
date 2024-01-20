const data = `[
  {
    "name": "Alice",
    "phone": "+1 234 567 8901"
  },
  {
    "name": "Bob",
    "phone": "+1 345 678 9012"
  },
  {
    "name": "John Galt",
    "phone": "+1 456 789 0123"
  },
  {
    "name": "Eve",
    "phone": "+1 567 890 1234"
  }
]`;
// Convertir les données JSON en un objet Javascript
const phoneList = JSON.parse(data);

// Parcourir la liste de numéros de téléphone avec une boucle for...of
for (const item of phoneList) {
  // Vérifier si le nom correspond à John Galt
  if (item.name === "John Galt") {
    // Afficher le numéro de téléphone de John Galt
    console.log("Le numéro de téléphone de John Galt est:", item.phone);
    // Arrêter la boucle
    break;
  }
}
