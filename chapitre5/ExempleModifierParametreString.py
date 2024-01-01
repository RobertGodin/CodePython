# -*- coding: utf-8 -*-
"""
Exemple modification de paramètre
"""

def f1(x):
    x[0]="d"
a="abc"
print("Valeur de a avant l'appel de f1:",a)
f1(a)  
print("Valeur de a après l'appel de f1:",a)