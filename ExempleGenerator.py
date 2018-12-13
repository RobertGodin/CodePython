# -*- coding: utf-8 -*-
def filtreSup0(seq) :
    for element in seq :
        if element > 0 :
            yield element
    
for element in filtreSup0([3,-1,4,6,0,8]):
	print(element)
