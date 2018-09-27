#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:57:14 2018

@author: nicolasvaldes

Solves P2 of Tarea 1, Metodos Numericos 2018 Semestre 2

Trabajar con datos de COBE de la radiacion de fondo cosmica. 

"""

import numpy as np
import matplotlib.pyplot as plt 
from astropy import constants as const

#CARGAR DATOS

x = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 0)
y = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 1)
err = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 3)

#GRAFICAR DATOS
fig, ax = plt.subplots() 
ax.errorbar(x * 3e10, y, xerr = 0, yerr = 400* err / 1000)

plt.xlabel("Frecuencia (s$^{-1}$)")
plt.ylabel("Espectro de Monopolo (MJy/sr)")

plt.savefig("espectro_monopolo.png",dpi=300)
plt.show()


#INTEGRAL POTENCIA; metodo trapecio
def integral(f,N,x_i = 0,x_f = np.pi / 2):
    ''' integra la funcion f usando metodo del trapecio. divide el intervalo en N partes.
        la integral por defecto va desde cero a pi/2. 
    '''
    h = 1 / N
    x_k = x_i + h 
    I = 0 
    for k in range(N):
        I = I + h / 2 * (f(x_k) + f(x_k + h))
        x_k = x_k + h 
    valor = I
    return valor 

def Integrando(x):
    return np.sin(x)**3 / (np.cos(x)**5 * (np.exp(np.tan(x))-1))

Integral = integral(Integrando,50)

P = 2 * const.h / (const.c ** 2) * (const.k_B / const.h) ** 4 * Integral

print(P)

print(Integral)


#INTEGRAR ESPECTRO


