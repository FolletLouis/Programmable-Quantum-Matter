import numpy as np
import matplotlib.pyplot as plt

def Cv(temp):

    N_avgdro = 6.022e23 
    kB = 1.380649e-23

    L = 5e-3 #m
    B = 5e-3 #m
    H = 5e-4 #m

    V = L*B*H
    temp = 0.1 #K

    h_sample = 0.084e-6 #W
    tau_th = 70e-6 #sec
    rho_diam = 3.52e3 #kg/m3
    M_diam = V * rho_diam #kg
    molar_mass_diam = 12.011e-3 # kg/mol
    moles_diam = M_diam / molar_mass_diam
    N_atoms = moles_diam * N_avgdro
    T_d = 2230 #K

    Cv = (12* ((np.pi)**4)/5) * ((temp/T_d)**3) * N_atoms * kB


    #constant = (h_sample * tau_th)/(8*Cv)

    return Cv