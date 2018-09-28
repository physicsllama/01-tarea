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
import scipy.integrate

#CARGAR DATOS

X = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 0)
Y = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 1)
err = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 3)

X = X * 3e10

np.insert(X,0,0)
np.insert(Y,0,0)

#GRAFICAR DATOS
fig, ax = plt.subplots() 
ax.errorbar(X, Y, xerr = 0, yerr = 400 * err / 1000)

plt.xlabel("Frecuencia (s$^{-1}$)")
plt.ylabel("Espectro de Monopolo (MJy/sr)")

plt.savefig("espectro_monopolo.png",dpi=300)
#plt.clf()


#INTEGRAL POTENCIA; metodo trapecio
def integral(f,N,x_i = 0,x_f = np.pi / 2):
    ''' integra la funcion f usando metodo del trapecio. divide el intervalo en N partes.
        la integral por defecto va desde cero a pi/2. 
    '''
    h = (x_f-x_i) / N
    x_k = x_i + h 
    I = 0 
    for k in range(N-1):
        I = I + h / 2 * (f(x_k) + f(x_k + h))
        x_k = x_k + h 
    valor = I
    return valor 

def Integrando(x):
    return np.sin(x)**3 / (np.cos(x)**5 * (np.exp(np.tan(x))-1))

Integral = integral(Integrando,100)

P = (2 * 6.626 * 1e-34 / ((3 * 1e8) ** 2)) * ((1.38 * 1e-23) / (6.626 * 1e-34)) ** 4 * Integral

print(P)


#INTEGRAR ESPECTRO
def integral_2():
    n = len(X)
    I = 0
    for k in range(n-1):
        I = I + (X[k+1]-X[k])/2 * (Y[k+1]+Y[k])
    return I * 1e-20 

print(integral_2())


T2 = (integral_2() / P) ** (1/4)

print(T2)

#GRAFICOS NUEVOS 
T1 = 2.725

def planck(T,v):
    B = (2 * const.h * v ** 3 / const.c ** 2) * 1e20 / (np.exp(6.626 * 1e-34 * v / (1.38 * 1e-23 *T))-1)
    return B

V = np.linspace(0,7e11,100)

plt.plot(V,planck(T1,V))
plt.plot(V,planck(T2,V))
plt.xlabel("Frecuencia (s$^{-1}$)")
plt.ylabel("Espectro de Monopolo (MJy/sr)")
plt.savefig("espectro2.png",dpi=300)
plt.show()



#PARTE 5, ALGORITMOS SCIPY
Integral_scipy = scipy.integrate.quad(Integrando,0,np.pi/2)[0]
Integral_2_scipy = scipy.integrate.trapz(Y,X) * 1e-20

R1 = Integral_scipy / Integral
R2 = Integral_2_scipy / integral_2()

print(R1)
print(R2)