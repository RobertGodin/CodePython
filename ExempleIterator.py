# -*- coding: utf-8 -*-
class SequenceSup0:
    """Représente la séquence des éléments supérieurs à 0"""
    def __init__(self, seq):
        self.seq = seq
        self.taille = len(seq)
        self.indice = 0	

    def __iter__(self):
        return self
    
    
    def __next__(self):
        while (self.indice < self.taille):
            if (self.seq[self.indice] > 0) :
                self.indice = self.indice + 1                
                return self.seq[self.indice-1]
            self.indice = self.indice + 1
        raise StopIteration

for element in SequenceSup0([3,-1,4,6,0,8]):
	print(element)
