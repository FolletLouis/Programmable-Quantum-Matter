"""
Heat capacity of diamond chip (from paper repo).
Used by Decoherence_sim, T2_heatmaps, Entanglement, Heat_Modeling.
NOTE: Original implementation uses fixed temp=0.1 K internally.
"""
import numpy as np


def Cv(temp: float) -> float:
    """
    Heat capacity of diamond chip based on Debye model.
    Parameters
    ----------
    temp : float
        Temperature (K). Note: original repo uses fixed 0.1 K internally.
    Returns
    -------
    float
        Heat capacity in J/K.
    """
    N_avgdro = 6.022e23
    kB = 1.380649e-23
    L, B, H = 5e-3, 5e-3, 5e-4  # m
    V = L * B * H
    temp = 0.1  # K - fixed in original for paper calculations
    rho_diam = 3.52e3  # kg/m3
    M_diam = V * rho_diam
    molar_mass_diam = 12.011e-3
    moles_diam = M_diam / molar_mass_diam
    N_atoms = moles_diam * N_avgdro
    T_d = 2230  # K
    Cv_val = (12 * (np.pi**4) / 5) * ((temp / T_d) ** 3) * N_atoms * kB
    return Cv_val
