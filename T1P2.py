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

x = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 0)
y = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 1)
err = np.loadtxt("firas_monopole_spec_v1.txt", usecols = 3)

fig, ax = plt.subplots() 
ax.errorbar(x * 3e11, y, xerr = 0, yerr = 400* err / 1000)

plt.xlabel("Frecuencia (s$^{-1}$)")
plt.ylabel("Espectro de Monopolo (MJy/sr)")

plt.savefig("espectro_monopolo.png",dpi=400)
plt.show()