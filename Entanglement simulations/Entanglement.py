import numpy as np
import matplotlib.pyplot as plt
from heat_capacity import *
from matplotlib.cm import plasma
import pandas as pd
from matplotlib.colors import to_hex



h_planck = 6.62607015e-34
kB = 1.380649e-23

# Parameters
t_pi_A = 150e-9      # pulse duration for A
t_pi_B = 10e-9       # pulse duration for B
#t_meas = 2e-4        # total measurement time
m = 121             # multiplier for B sequence
t_th = 70e-6         # thermalization timescale
T_cp = 100e-3        # baseline temp (K)

gamma = 1e-3

#N_values = np.array([2, 4, 8, 16])
h_sample_a = gamma * 0.084e-6  # W
h_sample_b = gamma * 0.207e-6 #W

# Fixed parameters
rise_rate = 5.2e-3  # fractional rise in temperature per uW
h_a = 26.34         # uW
h_b = 65            # uW


# Compute P(T) from formula
def P_temp_dependent(T, h_sample, t_pi):
    return ((9/8) * h_sample * t_pi) / (np.power(9,-1/8) * Cv(T))



#t must be an array
def Theta_SiV(t, t_meas, seq, N):

    #t = np.linspace(0, 8*t_meas, 100000)

    # Compute duty cycles
    dc_a = (t_pi_A * N) / t_meas
    dc_b = (t_pi_B * N * m) / t_meas

    # Compute effective heating powers
    h_a_eff = dc_a * h_a       # uW
    h_b_eff = dc_b * h_b       # uW

    # Compute fractional temperature rise
    frac_t_a = h_a_eff * rise_rate
    frac_t_b = h_b_eff * rise_rate

    # Absolute temperatures of cold plate stage
    T_df_a = (1 + frac_t_a) * T_cp
    T_df_b = (1 + frac_t_b) * T_cp


    F_total_A = np.zeros_like(t)
    F_total_B = np.zeros_like(t)

    if seq == "A" :
        # Pulse times for A
        t0_As = [(2 * j + 1) * t_meas / (2 * N) for j in range(N)]

        for t0 in t0_As:
            if t0 >= t[-1]:  # skip pulses outside the time window
                continue
            mask = t > t0
            delta_t = t[mask] - t0
            idx_closest = np.argmin(np.abs(t - t0))              # index of time closest to t0
            T_at_t0 = T_cp + F_total_A[idx_closest]
            P = P_temp_dependent(T_at_t0, h_sample_a, t_pi_A)
            F_total_A[mask] += P * (np.exp(-delta_t / t_th) - np.exp(-9 * delta_t / t_th))

        Temp_Siv_A = F_total_A + T_df_a 
        return Temp_Siv_A
    
    if seq == "B" :

        for j in range(N):
            t_j = (2 * j + 1) * t_meas / (2 * N)
            for k in range(1, m + 1):
                t0 = t_j + (k - 1) * t_pi_B
                if t0 >= t[-1]:  # ignore pulses beyond plot window
                    continue
                mask = t > t0
                delta_t = t[mask] - t0
                idx_closest = np.argmin(np.abs(t - t0))              # index of time closest to t0
                T_at_t0 = T_cp + F_total_A[idx_closest]
                P = P_temp_dependent(T_at_t0, h_sample_b, t_pi_B)
                F_total_B[mask] += P * (np.exp(-delta_t / t_th) - np.exp(-9 * delta_t / t_th))

        Temp_Siv_B = F_total_B + T_df_b 
        return Temp_Siv_B

#t_exp must be between t_meas and t_meas + t_cmpl and a number
def T2_SiV(t_exp, t_meas, N, seq, T2_orig):

    tau_range = np.linspace(0, t_exp, 10000)

    if seq == "A" :
        theta_vals_A = Theta_SiV(tau_range, t_meas, "A", N)
        T2_A_mod = T2_orig / (1 + T2_orig * (theta_vals_A[-1] - 0.1) * 3e6)
        return T2_A_mod
    
    if seq == "B" :
        theta_vals_B = Theta_SiV(tau_range, t_meas, "B", N)
        T2_B_mod = T2_orig / (1 + T2_orig * (theta_vals_B[-1] - 0.1) * 3e6)
        return T2_B_mod
    

#t_exp must be between t_meas and t_meas + t_cmpl and a number
def T1_SiV(t_exp, t_meas, N, seq, T1_orig):

    tau_range = np.linspace(0, t_exp, 10000)

    if seq == "A" :
        theta_vals_A = Theta_SiV(tau_range, t_meas, "A", N)
        T1_A_mod = T1_orig / (1 + T1_orig * (theta_vals_A[-1] - 0.1) * 2.4e6)
        return T1_A_mod
    
    if seq == "B" :
        theta_vals_B = Theta_SiV(tau_range, t_meas, "B", N)
        T1_B_mod = T1_orig / (1 + T1_orig * (theta_vals_B[-1] - 0.1) * 2.4e6)
        return T1_B_mod
    



#t_exp must be between t_meas and t_meas + t_cmpl and a number
def epsilon_jk(t_exp, t_meas, N, seq , T1_orig, T2_orig_j, T2_orig_k, zj, zk, F_dds):

    alpha_ent = 1e-4
    tau_range = np.linspace(0, t_exp, 10000)
    freq = 5e9 #Hz


    if seq == "A":
        theta_vals_A = Theta_SiV(tau_range, t_meas, "A", N)
        theta_SiV_A_exp = theta_vals_A[-1]

        p_th = 1/(1 + np.exp(-(h_planck * freq)/(kB * theta_SiV_A_exp)))
        

        T2_dyn_j = T2_SiV(t_exp, t_meas, N, "A", T2_orig_j)
        T2_dyn_k = T2_SiV(t_exp, t_meas, N, "A", T2_orig_k)
        T1_dyn = T1_SiV(t_exp, t_meas, N, "A", T1_orig)

        chi_prime_dep_j = np.power((t_exp/T2_dyn_j), zj) #- 0.5 * t_exp/T1_dyn
        chi_prime_dep_k = np.power((t_exp/T2_dyn_k), zk) #- 0.5 * t_exp/T1_dyn

   

        Fid_dep = (1 - alpha_ent * np.exp(-t_exp/T1_dyn)  - (1 - np.exp(-t_exp/T1_dyn))*p_th) * 0.5 * ( 1 + np.exp(-(chi_prime_dep_j + chi_prime_dep_k)) - np.sqrt(1 - np.exp(-2*chi_prime_dep_j)) * np.sqrt(1 - np.exp(-2*chi_prime_dep_k)))
        err_dep = (1 - Fid_dep)
        err_dds = (1-F_dds)* (N//2)


        err_tot = err_dep + err_dds

        return err_tot
    

    if seq == "B":

        theta_vals_B = Theta_SiV(tau_range, t_meas, "B", N)
        theta_SiV_B_exp = theta_vals_B[-1]
        p_th = 1/(1 + np.exp(-(h_planck * freq)/(kB * theta_SiV_B_exp)))
        T2_dyn_j = T2_SiV(t_exp, t_meas, N, "B", T2_orig_j)
        T2_dyn_k = T2_SiV(t_exp, t_meas, N, "B", T2_orig_k)
        T1_dyn = T1_SiV(t_exp, t_meas, N, "B", T1_orig)
        chi_prime_dep_j = np.power((t_exp/T2_dyn_j), zj) - 0.5 * t_exp/T1_dyn
        chi_prime_dep_k = np.power((t_exp/T2_dyn_k), zk) - 0.5 * t_exp/T1_dyn

        Fid_dep = (1 - alpha_ent * np.exp(-t_exp/T1_dyn)  - (1 - np.exp(-t_exp/T1_dyn))*p_th) * 0.5 * ( 1 + np.exp(-(chi_prime_dep_j + chi_prime_dep_k)) - np.sqrt(1 - np.exp(-2*chi_prime_dep_j)) * np.sqrt(1 - np.exp(-2*chi_prime_dep_k)))
        err_dep = (1 - Fid_dep)
        err_dds = (1-F_dds)* (N//2)

        err_tot = err_dep + err_dds

        return err_tot
    




def epsilon_jk_avg(t_cmpl, t_meas, N, seq , T1_orig, T2_orig_j, T2_orig_k, zj, zk, F_dds):

    delta_T = t_cmpl/100
    t_avg_array = np.linspace(t_meas, t_meas + t_cmpl, 100)
    sum = 0

    for t_exp in t_avg_array:
        eps = epsilon_jk(t_exp, t_meas, N, seq , T1_orig, T2_orig_j, T2_orig_k, zj, zk, F_dds)

        sum += eps * delta_T/t_cmpl


    return sum







    
















# def T2(tau, N):

#     T2(theta_siv, N)

#     theta_siv(tau)