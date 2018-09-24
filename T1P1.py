#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:31:31 2018

@author: nicolasvaldes

Solves P1 of Tarea 1, Metodos Numericos 2018 Semestre 2

Estimar la derivada de la funcion -cos(x) a orden h^4. 

"""

import numpy as np 
import matplotlib.pyplot as plt 

def derivada_1(f, x, step):
    g = (1 / h) * (f(x + h) - f(x))
    return g

def derivada_4(f, x, step):
    g = (1 / (12 * h)) * (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h))
    return g

x0 = 1.388 
h = np.logspace(-1, -15, 15, base = 10)

h = np.float64(h)

derivadas_1 = - derivada_1(np.cos, x0, h)
derivadas_4 = - derivada_4(np.cos, x0, h)

#derivadas_1 = np.float64(derivadas_1)
#derivadas_4 = np.float64(derivadas_4)

dif_1 = np.fabs(np.sin(x0) - derivadas_1)
dif_4 = np.fabs(np.sin(x0) - derivadas_4)

plt.xscale('log')
plt.yscale('log')
plt.plot(h, dif_1, label = "$\mathcal{O}(h)$")
plt.plot(h, dif_4, label = "$\mathcal{O}(h^4)$")
plt.legend()

plt.show()