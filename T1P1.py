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

#funciones utiles

def derivada_1(f, x, step):
    ''' toma una funcion f, calcula su derivada en x usando el metodo de orden h. step es el h.
    '''
    g = (1 / h) * (f(x + h) - f(x))
    return g

def derivada_4(f, x, step):
    '''toma una funcion f, calcula su derivada en x usando el metodo de orden h^4. step es el h.
    '''
    g = (1 / (12 * h)) * (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h))
    return g

#datos iniciales 

x0 = 1.388 
h = np.logspace(-1, -15, 15, base = 10)




#IMPLEMENTACION PARA FLOAT 32

#imponer tipo de float
x0 = np.float32(x0) 
h = np.float32(h)

#definir lista con derivadas de la funcion deseada
derivadas_1 = - derivada_1(np.cos, x0, h)
derivadas_4 = - derivada_4(np.cos, x0, h)

#definir error
dif_1 = np.fabs(np.sin(x0) - derivadas_1)
dif_4 = np.fabs(np.sin(x0) - derivadas_4)

#hacer graficos en escala logaritmica 
plt.xscale('log')
plt.yscale('log')
plt.plot(h, dif_1, label = "$\mathcal{O}(h)$")
plt.plot(h, dif_4, label = "$\mathcal{O}(h^4)$")
plt.legend()
plt.xlabel("Paso $h$")
plt.ylabel("Error $\Delta$")

plt.savefig("PlotP1_float32.png",dpi=300)  
plt.clf()





#IMPLEMENTACION PARA FLOAT 64

x0 = 1.388 
h = np.logspace(-1, -15, 15, base = 10)

x0 = np.float64(x0) 
h = np.float64(h)

derivadas_1 = - derivada_1(np.cos, x0, h)
derivadas_4 = - derivada_4(np.cos, x0, h)

dif_1 = np.fabs(np.sin(x0) - derivadas_1)
dif_4 = np.fabs(np.sin(x0) - derivadas_4)

plt.xscale('log')
plt.yscale('log')
plt.plot(h, dif_1, label = "$\mathcal{O}(h)$")
plt.plot(h, dif_4, label = "$\mathcal{O}(h^4)$")
plt.legend()
plt.xlabel("Paso $h$")
plt.ylabel("Error $\Delta$")

plt.savefig("PlotP1_float64.png",dpi=300)  
plt.clf()





#IMPLEMENTACION PARA FLOAT 128

x0 = 1.388 
h = np.logspace(-1, -15, 15, base = 10)

x0 = np.float128(x0) 
h = np.float128(h)

derivadas_1 = - derivada_1(np.cos, x0, h)
derivadas_4 = - derivada_4(np.cos, x0, h)

dif_1 = np.fabs(np.sin(x0) - derivadas_1)
dif_4 = np.fabs(np.sin(x0) - derivadas_4)

plt.xscale('log')
plt.yscale('log')
plt.plot(h, dif_1, label = "$\mathcal{O}(h)$")
plt.plot(h, dif_4, label = "$\mathcal{O}(h^4)$")
plt.legend()
plt.xlabel("Paso $h$")
plt.ylabel("Error $\Delta$")

plt.savefig("PlotP1_float128.png",dpi=300)  
plt.clf()