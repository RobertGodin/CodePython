# -*- coding: utf-8 -*-
"""
Exemple classe Contact
Variable d'objet publique
"""

class Contact :
    """ Un objet représente un contact 

        nom : str
        prenom : str
        numero_telephone : str
    """

    def __init__(self,nom,prenom,numero_telephone):
        self.nom = nom
        self.prenom = prenom
        self.numero_telephone = numero_telephone
        
    def __str__(self):
        return 'Le numéro de téléphone de '+self.prenom+' '+self.nom+' est :'+self.numero_telephone
    

contact1 = Contact('Binette','Bob','333-333-3333')
print(contact1)
contact1.numero_telephone = '444-444-4444'
print(contact1)
contact1.age= 52
print(contact1)

