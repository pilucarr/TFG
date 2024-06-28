#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

random.seed(2)
np.random.seed(2)

# Funciones del modelo

def main(V, t, p, Ie, Ina17, Ina18, IKDR, IKa, Il, params):
    C = params[0]
    A = params[1]
    
    if t < 50 or t > 60:
        Ie = 0
    
    dVdt = (1/C) * (Ie/A - (Ina17 + Ina18 + IKDR + IKa + Il))
    
    return dVdt

# A continuación se calcula la intensidad que recorre cada canal

def fIna17(g17, m17, h17, s17, Ena, V):
    Ina17 = g17*(m17**3)*h17*s17*(V - Ena)
    return Ina17

def fIna18(g18, m18, h18, Ena, V):
    Ina18 = g18*m18*h18*(V - Ena)
    return Ina18

def fIKDR(gKDR, nKDR, Ek, V):
    IKDR = gKDR*nKDR*(V - Ek)
    return IKDR

def fIKa(gKa, nKa, hKa, Ek, V):
    IKa = gKa*nKa*hKa*(V - Ek)
    return IKa

def fIl(gl, El, V):
    Il = gl*(V - El)
    return Il

# Cálculo de las variables de activación e inactivación de cada canal
# CANAL Na 17 -- 24 constantes -- p[0:23]
# p[0:7]
def fm17(m17, t, p, V):
    a_m17 = p[0] + ((p[1])/(1 + math.exp(-(V + p[2])/p[3])))
    b_m17 = p[4] + ((p[5])/(1 + math.exp((V + p[6])/p[7])))
    
    m17_inf = (a_m17)/(a_m17 + b_m17)
    tau_m17 = 1/(a_m17 + b_m17)
    
    dm17dt = (m17_inf - m17)/tau_m17
        
    return dm17dt

# p[8:15]
def fh17(h17,t, p, V): 
    a_h17 = p[0] + ((p[1])/(1 + math.exp((V + p[2])/p[3])))
    b_h17 = p[4] + ((p[5])/(1 + math.exp(-(V + p[6])/p[7])))
    
    h17_inf = (a_h17)/(a_h17 + b_h17)
    tau_h17 = 1/(a_h17 + b_h17)
    
    dh17dt = (h17_inf - h17)/tau_h17
    
    return dh17dt

# p[16:23]
def fs17(s17, t, p, V):
    a_s17 = p[0] + ((p[1])/(1 + math.exp((V + p[2])/p[3])))
    b_s17 = p[4] + ((p[5])/(1 + math.exp((V + p[6])/p[7])))
    
    s17_inf = (a_s17)/(a_s17 + b_s17)
    tau_s17 = 1/(a_s17 + b_s17)
   
    ds17dt = (s17_inf - s17)/tau_s17
        
    return ds17dt

# CANAL Na 18 -- 14 constantes -- p[24:37]
# p[24:31]
def fm18(m18, t, p, V):
    a_m18 = p[0] + ((p[1])/(1 + math.exp((V + p[2])/p[3])))
    b_m18 = p[4] + ((p[5])/(1 + math.exp((V + p[6])/p[7])))
   
    m18_inf = (a_m18)/(a_m18 + b_m18)
    tau_m18 = 1/(a_m18 + b_m18)
    
    dm18dt = (m18_inf - m18)/tau_m18
    
    return dm18dt

# p[32:37]
def fh18(h18, t, p, V):    
    h18_inf = (1/(1 + math.exp((V + p[0])/p[1])))
    tau_h18 = p[2] + p[3]*math.exp(-(V + p[4])**2/(2*p[5]**2))
    
    dh18dt = (h18_inf - h18)/tau_h18
        
    return dh18dt

# CANAL KDR -- 9 constantes -- p[38:46]
def fnKDR(nKDR, t, p, V):
     if V == -14.273:
         a_nKDR = p[0]*10
     else:
         a_nKDR = (p[0]*(V + p[1]))/(1 - math.exp(-(V + p[2])/p[3]))
     b_nKDR = p[4]*math.exp(-(V + p[5])/p[6])
     
     nKDR_inf = 1/(1 + math.exp(-(V + p[7])/p[8]))
     tau_nKDR = 1/(a_nKDR + b_nKDR) + 1
     
     dnKDRdt = (nKDR_inf - nKDR)/tau_nKDR
         
     return dnKDRdt

# CANAL Ka -- 12 constantes -- p[47:58]
# p[47:52]
def fnKa(nKa, t, p, V): 
    nKa_inf = (1/(1 + math.exp(-(V + p[0])/p[1])))**4
    tau_nKa = p[2] + p[3]*math.exp((-(V + p[4])**2)/(2*p[5]**2))
    
    dnKadt = (nKa_inf - nKa)/tau_nKa
    
    return dnKadt

# p[53:58]
def fhKa(hKa, t, p, V):
    hKa_inf = 1/(1 + math.exp((V + p[0])/p[1]))
    tau_hKa = p[2] + p[3]*math.exp((-(V + p[4])**2)/(2*p[5]**2))
    if tau_hKa < 5:
        tau_hKa = 5
    
    dhKadt = (hKa_inf - hKa)/tau_hKa
        
    return dhKadt


def simulate_model(p, I_ext, V0, params):
    # Constantes
    Ena = params[2]
    Ek = params[3]
    El = params[4]
    g17 = params[5]
    g18 = params[6]
    gKDR = params[7]
    gKa = params[8]
    gl = params[9]


    # Condiciones iniciales
    def val_ini(V0,p):
        
        a_m17 = p[0] + ((p[1])/(1 + math.exp(-(V0 + p[2])/p[3])))
        b_m17 = p[4] + ((p[5])/(1 + math.exp((V0 + p[6])/p[7])))
        
        m17 = (a_m17)/(a_m17 + b_m17)
        
        a_h17 = p[8] + ((p[9])/(1 + math.exp((V0 + p[10])/p[11])))
        b_h17 = p[12] + ((p[13])/(1 + math.exp(-(V0 + p[14])/p[15])))
        
        h17 = (a_h17)/(a_h17 + b_h17)
        
        a_s17 = p[16] + ((p[17])/(1 + math.exp((V0 + p[18])/p[19])))
        b_s17 = p[20] + ((p[21])/(1 + math.exp((V0 + p[22])/p[23])))
        
        s17 = (a_s17)/(a_s17 + b_s17)
        
        a_m18 = p[24] + ((p[25])/(1 + math.exp((V0 + p[26])/p[27])))
        b_m18 = p[28] + ((p[29])/(1 + math.exp((V0 + p[30])/p[31])))
       
        m18 = (a_m18)/(a_m18 + b_m18)
        
        h18 = (1/(1 + math.exp((V0 + p[32])/p[33])))
        
        nKDR = 1/(1 + math.exp(-(V0 + p[45])/p[46]))
        
        nKa = (1/(1 + math.exp(-(V0 + p[47])/p[49])))**4
        
        hKa = 1/(1 + math.exp((V0 + p[53])/p[54]))
        
        return m17, h17, s17, m18, h18, nKDR, nKa, hKa
    # Valores iniciales:
    V = V0
    m17, h17, s17, m18, h18, nKDR, nKa, hKa = val_ini(V, p)
        
    #Recoger valores para representar.
    Vrec = [V]
    t = 0
    # Bucle para la evolución temporal
    for i in range(0,3199):
        t = t + 0.05
        # Cálculo variables activación-inactivación
        # Na17
        m17 = m17 + 0.05*fm17(m17, t, p[0:8], V)
       
        h17 = h17 + 0.05*fh17(h17, t, p[8:16], V)
       
        s17 = s17 + 0.05*fs17(s17, t, p[16:24], V)
        
        # Na18
        m18 = m18 + 0.05*fm18(m18, t, p[24:32], V)
       
        h18 = h18 + 0.05*fh18(h18, t, p[32:38], V)
        
        # KDR
        nKDR = nKDR + 0.05*fnKDR(nKDR, t, p[38:47], V)
        
        # Ka
        nKa = nKa + 0.05*fnKa(nKa, t, p[47:53], V)
       
        hKa = hKa + 0.05*fhKa(hKa, t, p[53:], V)
        
        v = [m17,h17,s17,m18,h18,nKDR,nKa,hKa]
        
        for k in range(len(v)):
            if v[k]<0 or v[k]>1:
                raise Exception("Out of range")
        
        # Cálculo corrientes
        Ina17 = fIna17(g17, m17, h17, s17, Ena, V)
        Ina18 = fIna18(g18, m18, h18, Ena, V)
        IKDR = fIKDR(gKDR, nKDR, Ek, V)
        IKa = fIKa(gKa, nKa, hKa, Ek, V)
        Il = fIl(gl, El, V)
        # Cálculo potencial
        V = V + 0.05*main(V, t , p, I_ext, Ina17, Ina18, IKDR, IKa, Il, params)
        
        Vrec.append(V)
    return np.array(Vrec)

def createindv(p_size, p_bounds, ind_train): 
    fitness = 1000000000000000000000000000000000000000
    while fitness == 1000000000000000000000000000000000000000:
        indv = []
        for i in range(p_size):
            bounds = p_bounds[i]
            a,b = bounds
            indv.append(np.random.uniform(a,b))
        
        fitness = func_to_optimize(indv,ind_train)
    return indv


# Parámetros del problema
p_size = 59  # Tamaño del vector p 
"""p_bounds = [
    (-5, 5), (-30, 30), (-15, 5), (-10, 20), 
    (-5,5), (-10,50), (-5, 100), (-5,30),
    (-5,5), (-5,5), (90,170), (-5,30),
    (-5,5), (-5, 20), (-5, 20), (-5, 30),
    (-5,5), (-5,5), (70,150), (-5, 40),
    (100,170), (-170,-100), (-500,-300), (-5,50),
    (-10,10), (-10,10), (-10,10), (-5, 30),
    (-5,5), (-20,20), (-5, 70), (-20,20),
    (-20, 50), (-10,10),
    (-10,10), (-5, 70), (-5,70), (-10,30),
    (-5,5), (-30,30), (-30,30), (-5, 30),
    (-5,5), (-5,70), (-10,10),
    (-5,30), (-5,30),
    (-10,20), (-5,30),
    (-5,5), (-20,20), (-30,40), (-5, 50),
    (-5, 70), (-10, 10),
    (-10, 30), (10, 70), (10, 60), (10,60)
]"""
p_bounds = [
    (-0.01, 0.01), (14.5, 16.5), (-7, -4), (10, 14), 
    (-0.01, 0.01), (34.2,36), (70, 75), (15,17.5),
    (-0.01,0.01), (-1,1), (120,125), (14,16),
    (-0.01,0.01), (0, 4), (3, 7), (10, 14),
    (-0.1,0.1), (-0.1, 0.1), (90,95), (15, 17),
    (130,135), (-135,-130), (-390,-380), (20,35),
    (0,5), (-5,1), (-5,1), (10, 15),
    (-0.01,0.01), (5,10), (40, 60), (5,10),
    (25, 40), (0, 10),
    (-1, 5), (30, 50), (25, 50), (10, 20),
    (-0.1, 0.1), (10, 20), (10, 20), (5, 15),
    (-1,1), (50,60), (0,5),
    (10,20), (15,35),
    (0,10), (10,20),
    (-1,1), (7,17), (20, 30), (30, 40),
    (40, 60), (0, 10),
    (10, 30), (40, 60), (30, 50), (30, 50)
]

# Importamos datos reales
Tot_Iext= (pd.read_csv('Datos_Control_reobase.csv', delimiter=';', header=0).to_numpy())
control = pd.read_csv('Datos_Control_APs_IB4_.csv', delimiter=';',
                      header=1,keep_default_na=False)

I_ext = np.delete(Tot_Iext,[5,13,21])
indices = [6,14,22,28,29,30,31,32,33,34]
columns_to_drop = control.columns[indices]
Tot_datos_control = control.drop(columns_to_drop, axis=1).to_numpy()

ind_test = [7,14,20]
ind_train = [4,5,10,12,13,15,19,21,22]

# Datos reales usados para el ajuste (selección de neurona)
C = 0.93
A = 21.68
Ena = 67.10
Ek = -84.70
El = -58.91
g17 = 18
g18 = 7
gKDR = 4.78
gKa = 8.33
gl = 0.0575

params = [C, A, Ena, Ek, El, g17, g18, gKDR, gKa, gl]

def func_to_optimize(indv,ind_train):
    try:
        tot_error = 0
        for ind in ind_train:
            Ie = I_ext[ind-1]  
            V0 = Tot_datos_control[0,ind]  
            real_data = Tot_datos_control[:,ind] 
            Vrec = simulate_model(indv, Ie, V0, params)
            error = np.mean((Vrec - real_data)**2)
           # dist_peak = abs( max(Vrec)-max(real_data))
            tot_error = tot_error + error 

    except:
        tot_error = 1000000000000000000000000000000000000000
    return tot_error

"""
best= [0, 15.5, -5, 12.08, 0, 35.2, 72.7, 16.7, 0, 0.38685, 122.35, 15.29,
     -0.00283, 2.00283, 5.5266, 12.70195, 0.00003, 0.00092, 93.9, 16.6,
     132.05, -132.05, -384.9, 28.5, 2.85, -2.839, -1.159, 13.95, 0, 7.6205,
     46.463, 8.8289, 32.2, 4, 1.218, 42.043, 38.1, 15.19, 0.001265, 14.273,
     14.273, 10, 0.125, 55, 2.5, 14.62, 18.38, 5.4, 16.4, 0.25,
     10.04, 24.67, 34.8, 49.9, 4.6, 20, 50, 40, 40]
"""
best = createindv(p_size, p_bounds, ind_train)

fitness_best = func_to_optimize(best,ind_train)
print(fitness_best)

for n in range(1000):
    p = createindv(p_size, p_bounds, ind_train)
    fitness = func_to_optimize(p,ind_train)
    if fitness<fitness_best:
        best = p
        fitness_best = fitness

print(best)
print(fitness_best)

Ie = I_ext[13]  
V0 = Tot_datos_control[0,14]  
tp = Tot_datos_control[:,0]  
real_data = Tot_datos_control[:,14]  

Vmodel = simulate_model(best, Ie, V0, params)

plt.plot(tp,Vmodel,tp,real_data)
plt.show()
#error = genetic_obj._fitness_function(hijo["Individual"])


