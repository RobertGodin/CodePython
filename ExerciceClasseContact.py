# -*- coding: utf-8 -*-
"""
Exercice classe Contact
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
        return 'Le numero de téléphone de '+self.prenom+' '+self.nom+' est :'+self.numero_telephone
    
liste_contacts = []
liste_contacts.append(Contact('Binette','Bob','333-333-3333'))
liste_contacts.append(Contact('Emerson','Keith','111-111-1111'))
liste_contacts.append(Contact('Anderson','Ian','222-222-2222'))

for un_contact in liste_contacts:
    print(un_contact)