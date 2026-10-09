# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:23:54 2021

@author: vango
"""

def recherche_lineaire(element, liste):
    """ Recherche linéaire. 
        Retourne vrai si element est dans la liste, faux sinon
    """
    for indice in range(len(liste)):
        if liste[indice] == element:
            return True
    return False

liste = [ 7,5,10,1,25,8,20 ]
element = 8
print("------------- Recherche linéaire de ",element," dans ", liste)
if recherche_lineaire(element, liste):
    print("Element présent")
else:
    print("Element absent")

liste = [ 7,5,10,1,25,8,20 ]
element = 8
print("------------- Recherche linéaire de ",element," dans ", liste)
if recherche_lineaire(element, liste):
    print("Element présent")
else:
    print("Element absent")


def recherche_lineaire_triee(element, liste_triee):
    """ Recherche linéaire améloirée. Suppose que la liste est triée.
        Retourne vrai si element est dans la liste, faux sinon
    """
    for indice in range(len(liste_triee)):
        if liste_triee[indice] == element:
            return True
        if liste_triee[indice] > element:
            return False
    return False

liste = [ 7,5,10,1,25,8,20 ]
liste.sort()
element = 8
print("------------- Recherche linéaire améliorée de ",element," dans liste triée", liste)
if recherche_lineaire_triee(element, liste):
    print("Element présent")
else:
    print("Element absent")

def recherche_binaire(element, liste_triee):
    """ Recherche binaire. Suppose que la liste est triée.
        Retourne vrai si element est dans la liste, faux sinon
    """
    borne_inferieure = 0
    borne_superieure = len(liste_triee) - 1
    milieu = 0
 
    while borne_inferieure <= borne_superieure:
        milieu = (borne_superieure + borne_inferieure) // 2
        if liste_triee[milieu] < element:
            borne_inferieure = milieu + 1
        elif liste_triee[milieu] > element:
            borne_superieure = milieu - 1
        else: # liste_triee[milieu] == element 
            return True
    return False

liste = [ 7,5,10,1,25,8,20 ]
liste.sort()
element = 8
print("------------- Recherche binaire de ",element," dans liste triée", liste)
if recherche_binaire(element, liste):
    print("Element présent")
else:
    print("Element absent")

def recherche_indice_binaire(element,liste_triee):
    """ Recherche binaire. Suppose que la liste est triée.
        Retourne l'indice de l'element si dans la liste, -1 sinon
    """
    borne_inferieure = 0
    borne_superieure = len(liste_triee) - 1
    milieu = 0
 
    while borne_inferieure <= borne_superieure:
        milieu = (borne_superieure + borne_inferieure) // 2
        if liste_triee[milieu] < element:
            borne_inferieure = milieu + 1
        elif liste_triee[milieu] > element:
            borne_superieure = milieu - 1
        else: # liste_triee[milieu] == element 
            return milieu
    return -1

liste = [ 7,5,10,1,25,8,20 ]
liste.sort()
element = 8
print("------------- Recherche binaire de l'indice de ",element," dans liste triée", liste)
print("Indice de l'élément (-1 si absent):",recherche_indice_binaire(element,liste))

def recherche_binaire_recursive(element,liste_triee, borne_inferieure, borne_superieure):
    """ Recherche binaire récursive. Suppose que la liste est triée.
        Retourne vrai si element est dans la liste, faux sinon
    """

    if borne_superieure >= borne_inferieure:
        milieu = (borne_superieure + borne_inferieure) // 2
        if liste_triee[milieu] == element:
            return True
        elif liste_triee[milieu] > element:
            return recherche_binaire_recursive(liste_triee, borne_inferieure, milieu - 1, element)
        else:
            return recherche_binaire_recursive(liste_triee, milieu + 1, borne_superieure, element)
    else:
        # L'élément est absent
        return False

liste = [ 7,5,10,1,25,8,20 ]
liste.sort()
element = 8
print("------------- Recherche binaire récursive de ",element," dans liste triée", liste)
if recherche_binaire_recursive(element, liste,0,len(liste)):
    print("Element présent")
else:
    print("Element absent")

def recherche_binaire_recursive_indice(element,liste_triee,borne_inferieure, borne_superieure):
    """ Recherche binaire récursive. Suppose que la liste est triée.
        Retourne l'indice de l'element si dans la liste, -1 sinon
    """
    if borne_superieure >= borne_inferieure:
 
        milieu = (borne_superieure + borne_inferieure) // 2
 
        # If element is present at the milieudle itself
        if liste_triee[milieu] == element:
            return milieu
 
        # If element is smaller than milieu, then it can only
        # be present in left subliste_trieeay
        elif liste_triee[milieu] > element:
            return recherche_binaire_recursive(element,liste_triee, borne_inferieure, milieu - 1)
 
        # Else the element can only be present in right subliste_trieeay
        else:
            return recherche_binaire_recursive(element,liste_triee, milieu + 1, borne_superieure, element)
 
    else:
        # Element is not present in the liste_trieeay
        return -1
 
liste = [ 7,5,10,1,25,8,20 ]
liste.sort()
element = 8
print("------------- Recherche binaire récursive de l'indice de ",element," dans liste triée", liste)
print("Indice de l'élément (-1 si absent):",recherche_binaire_recursive_indice(element,liste,0,len(liste)))