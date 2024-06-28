#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:13:10 2024

@author: caarp
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy 
import math
from scipy.integrate import odeint


# imprimir con notación matemática.
sympy.init_printing(use_latex='mathjax')

# Módulo funciones ------------------------------------------------------------
def main(V, t, p, Ie, Ina17, Ina18, IKDR, IKa, Il):
    # Definimos las constantes que necesitamos en unidades del SI*
    C = 0.93
    A = 21.68
    
    if t<50 or t>60:
    #if t<50:
        
        Ie = 0
        
    # Se recoge la función a resolver, es igual a la primera derivada del
    # potencial respecto del tiempo

    dVdt = (1/C)*(Ie/A - (Ina17 + Ina18 + IKDR + IKa +Il))
    #if dVdt > 0:
     #   print(dVdt,t)
        
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
         a_nKDR = (p[0]*(V + p[1]))/(1 - 
                                     math.exp(-(V + p[2])/p[3]))
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
    m17rec = [m17]
    h17rec = [h17]
    s17rec = [s17]
    m18rec = [m18]
    h18rec = [h18]
    nKDRrec = [nKDR]
    nKarec = [nKa]
    hKarec = [hKa]
    Ina17rec = []
    Ina18rec = []
    IKDRrec = []
    IKarec = []
    Ilrec = []
    t = 0
    # Bucle para la evolución temporal
    for i in range(0,3199):
        t = Tot_datos_control[i,0]
        # Cálculo variables activación-inactivación
        # Na17
        m17 = m17 + 0.05*fm17(m17, t, p[0:8], V)
        m17rec.append(m17)
        h17 = h17 + 0.05*fh17(h17, t, p[8:16], V)
        h17rec.append(h17)
        s17 = s17 + 0.05*fs17(s17, t, p[16:24], V)
        s17rec.append(s17)
        # Na18
        m18 = m18 + 0.05*fm18(m18, t, p[24:32], V)
        m18rec.append(m18)
        h18 = h18 + 0.05*fh18(h18, t, p[32:38], V)
        h18rec.append(h18)
        # KDR
        nKDR = nKDR + 0.05*fnKDR(nKDR, t, p[38:47], V)
        nKDRrec.append(nKDR)
        # Ka
        nKa = nKa + 0.05*fnKa(nKa, t, p[47:53], V)
        nKarec.append(nKa)
        hKa = hKa + 0.05*fhKa(hKa, t, p[53:], V)
        hKarec.append(hKa)
        #print(i)
        # Cálculo corrientes
        Ina17 = fIna17(g17, m17, h17, s17, Ena, V)
        Ina17rec.append(Ina17)
        Ina18 = fIna18(g18, m18, h18, Ena, V)
        Ina18rec.append(Ina18)
        IKDR = fIKDR(gKDR, nKDR, Ek, V)
        IKDRrec.append(IKDR)
        IKa = fIKa(gKa, nKa, hKa, Ek, V)
        IKarec.append(IKa)
        Il = fIl(gl, El, V)
        Ilrec.append(Il)
        # Cálculo potencial
        V = V + 0.05*main(V, t , p, I_ext, Ina17, Ina18, IKDR, 
                          IKa, Il)
        Vrec.append(V)
        
    return Vrec, m17rec, h17rec, s17rec, m18rec, h18rec, nKDRrec, nKarec, hKarec, Ina17rec, Ina18rec, IKDRrec, IKarec

def fitness(p,params):
    fitness = 0
    ind_test = [7,14,20]
    for n in ind_test:
        Ie = I_ext[n-1]  
        V0 = Tot_datos_control[0,n]  
        real_data = Tot_datos_control[:,n] 
        Vrec = simulate_model(p, Ie, V0, params)
        error = np.mean((Vrec[0] - real_data)**2)
        fitness = fitness + error
        
    fitness = fitness/len(ind_test)
    return fitness

#------------------------------------------------------------------------------
"""
p = [0, 15.5, -5, 12.08, 0, 35.2, 72.7, 16.7, 0, 0.38685, 122.35, 15.29,
     -0.00283, 2.00283, 5.5266, 12.70195, 0.00003, 0.00092, 93.9, 16.6,
     132.05, -132.05, -384.9, 28.5, 2.85, -2.839, -1.159, 13.95, 0, 7.6205,
     46.463, 8.8289, 32.2, 4, 1.218, 42.043, 38.1, 15.19, 0.001265, 14.273,
     14.273, 10, 0.125, 55, 2.5, 14.62, 18.38, 5.4, 16.4, 0.25,
     10.04, 24.67, 34.8, 49.9, 4.6, 20, 50, 40, 40]
"""

p=[0, 16.21776647815727, -4.168588586522469, 13.58789735485021, -0.0027540478592733494, 34.48536300316938, 73.20204123776823, 16.191831163530093, 0.007496375314406346, 0.15444380514184708, 122.52548600232285, 14.229137399396311, 0.12501902631021, 2.531903619516132, 6.937400980504057, 13.941686608146522, 3e-05, 0.09679575932449516, 90.61051592503213, 15.539508405064995, 131.329457642136, -130.82233648556237, -389.0256420038235, 34.95348335905072, 2.870113676447812, -2.839, 0.23660214776920796, 13.95, -0.0025035250942871956, 7.450819239690222, 46.463, 8.8289, 32.8558892651739, 3.8619294497287804, 1.046412284211998, 42.043, 31.677122653454067, 15.156182819220984, 0.095674413527461, 14.831280168750226, 14.273, 12.204773697971465, 0.125, 55, 2.5, 14.62, 18.38, 9.941365893452828, 19.754403498216355, 0.1272746847767492, 11.124261923792226, 28.76143795507063, 36.34696984314308, 45.94797249846325, 2.5417242379554947, 23.808792208913868, 42.08439567452904, 41.76054360462657, 47.79266548246906]


Tot_Iext= (pd.read_csv('Datos_Control_reobase.csv', delimiter=';', header=0).to_numpy())
control = pd.read_csv('Datos_Control_APs_IB4_.csv', delimiter=';',
                      header=1,keep_default_na=False)

I_ext = np.delete(Tot_Iext,[5,13,21])
indices = [6,14,22,28,29,30,31,32,33,34]
columns_to_drop = control.columns[indices]
Tot_datos_control = control.drop(columns_to_drop, axis=1).to_numpy()

ind_test = [7,14,20]
ind_train = [4,5,10,12,13,15,19,21,22]
# Cambiar índice segun cual quieras usar de valores iniciales
indice = 14

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
   


fitness_test = fitness(p, params)
print(fitness_test)

Ie = I_ext[indice-1] 
#Ie = 100
V0 = Tot_datos_control[0,indice] 
#V0 = -66.47794383085288
#V0 = -58.49244311213793
Vrec, m17rec, h17rec, s17rec, m18rec, h18rec, nKDRrec, nKarec, hKarec, Ina17rec, Ina18rec, IKDRrec, IKarec = simulate_model(p, Ie, V0, params)

# Gráficas, ir descomentando según se quiera representar
tp = Tot_datos_control[:,0]
  
#plt.plot(tp,Vrec, label = 'Modelo')

#for i in ind_train:
#    plt.plot(tp,Tot_datos_control[:,i],linewidth=1)
#for i in ind_test:
 #   plt.plot(tp,Tot_datos_control[:,i],linewidth=1)
    
plt.plot(tp,Tot_datos_control[:,20],label = 'Test 3',linewidth=1)
plt.plot(tp,Vrec,'r' ,label = 'Modelo',linewidth=1)
#plt.plot(tp,Tot_datos_control[:,20],linewidth=1)
#plt.xlim(40,80)
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.legend()
# Guardar el gráfico con buena resolución
plt.savefig('test3.jpg', dpi=600)
plt.show()
"""
plt.plot(tp,Vrec, label = 'Modelo')
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.annotate(r'$I_{ext} = 100 pA$', xy = (20,40))
plt.xlim(0,30)
# Guardar el gráfico con buena resolución
plt.savefig('modelo.jpg', dpi=600)
plt.show()

plt.plot(tp,Vrec, label = 'Modelo',linewidth=1)
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.legend()
plt.plot(tp,datos_control[:,14], 'r', label = 'Datos reales',linewidth=1)
#plt.xlim(40,80)
plt.legend()
# Guardar el gráfico con buena resolución
plt.savefig('comparativaa.jpg', dpi=600)
plt.show()

plt.plot(tp, m17rec, label = r'$m_{17}$')
plt.plot(tp, h17rec, label = r'$h_{17}$') 
plt.plot(tp, s17rec, label = r'$s_{17}$')
plt.plot(tp, m18rec, label = r'$m_{18}$')
plt.plot(tp, h18rec, label = r'$h_{18}$')
plt.plot(tp, nKDRrec, label = r'$n_{KDR}$')
plt.plot(tp, nKarec, label = r'$n_{Ka}$')
plt.plot(tp, hKarec, label = r'$h_{Ka}$')
plt.xlabel('t (ms)')
plt.xlim(40,80)
plt.legend()
# Guardar el gráfico con buena resolución
plt.savefig('variables_LS.jpg', dpi=600)
plt.show()

tp2 = Tot_datos_control[:3199,0]
#plt.plot(tp,Vrec,'--' ,label = 'Modelo')
plt.plot(tp2, Ina17rec, label = r'$I_{Na1.7}$')
plt.plot(tp2, Ina18rec, label = r'$I_{Na1.8}$') 
plt.plot(tp2, IKDRrec, label = r'$I_{KDR}$')
plt.plot(tp2, IKarec, label = r'$I_{KA}$')
#plt.plot(tp2, Ilrec, label = r'$I_{l}$')
plt.ylabel('I (pA)')
plt.xlabel('t (ms)')
plt.xlim(40,80)
plt.legend()
# Guardar el gráfico con buena resolución
plt.savefig('corrientes_LS.jpg', dpi=600)
plt.show()

plt.plot(tp,datos_control[:,1:], linewidth=0.8)
#plt.xlim(40,80)
#plt.ylim(0,1)
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.savefig('total_datos.jpg', dpi=600)
#plt.savefig('Grafica.jpg', dpi=600)
plt.show()
"""
