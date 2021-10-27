# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:18:46 2021

@author: vango
"""
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(1,11,1)
plt.plot(x,np.log(x), label="y=ln(x)")
plt.plot(x,x, label="y=x")
plt.plot(x,x*np.log(x), label="y=xln(x)")
plt.plot(x,x**2,label = "y=x*x")
plt.legend(loc='upper left')
plt.show()