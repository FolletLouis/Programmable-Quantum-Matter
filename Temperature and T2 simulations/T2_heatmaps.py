import numpy as np
import matplotlib.pyplot as plt
from heat_capacity import *
from matplotlib.cm import plasma
import pandas as pd
from matplotlib.colors import LogNorm



# Parameters
t_pi_A = 150e-9      # pulse duration for A
t_pi_B = 10e-9       # pulse duration for B
t_meas = 8e-4        # total measurement time
m = 121             # multiplier for B sequence
t_th = 70e-6         # thermalization timescale
T_cp = 100e-3        # baseline temp (K)

gamma_mech = 1e-3

N_values = np.array([2, 4, 8, 16])
h_sample_a = gamma_mech * 0.084e-6  # W
h_sample_b = gamma_mech * 0.207e-6 #W

# Time array
t = np.linspace(0, 2*t_meas, 100000)

# Fixed parameters
rise_rate = 5.2e-3  # fractional rise in temperature per uW
h_a = 26.34         # uW
h_b = 65            # uW

t_meas_values = [t_meas] 


colors = plasma(np.linspace(0, 0.8, 2))

#Dil Fridge
for idx, t_meas in enumerate(t_meas_values):
    # Compute duty cycles
    dc_a_array = (t_pi_A * N_values) / t_meas
    dc_b_array = (t_pi_B * N_values * m) / t_meas

    # Compute effective heating powers
    h_a_eff_array = dc_a_array * h_a       # uW
    h_b_eff_array = dc_b_array * h_b       # uW

    # Compute fractional temperature rise
    frac_t_a_array = h_a_eff_array * rise_rate
    frac_t_b_array = h_b_eff_array * rise_rate

    # Absolute temperatures
    T_a_array = (1 + frac_t_a_array) * T_cp
    T_b_array = (1 + frac_t_b_array) * T_cp



Temp_A_list = []
Temp_B_list = []
# Compute P(T) from formula
def P_temp_dependent(T, h_sample, t_pi):
    return ((9/8) * h_sample * t_pi) / (np.power(9,-1/8) * Cv(T))



for N in N_values:
    F_total_A = np.zeros_like(t)
    F_total_B = np.zeros_like(t)

    # Pulse times for A
    t0_As = [(2 * j + 1) * t_meas / (2 * N) for j in range(N)]

    for t0 in t0_As:
        mask = t > t0
        delta_t = t[mask] - t0
        idx_closest = np.argmin(np.abs(t - t0))              # index of time closest to t0
        T_at_t0 = T_cp + F_total_A[idx_closest]
        P = P_temp_dependent(T_at_t0, h_sample_a, t_pi_A)
        F_total_A[mask] += P * (np.exp(-delta_t / t_th) - np.exp(-9 * delta_t / t_th))

    # Pulse times for B (nested)
    for j in range(N):
        t_j = (2 * j + 1) * t_meas / (2 * N)
        for k in range(1, m + 1):
            t0 = t_j + (k - 1) * t_pi_B
            if t0 >= t[-1]:  # ignore pulses beyond plot window
                continue
            mask = t > t0
            delta_t = t[mask] - t0
            idx_closest = np.argmin(np.abs(t - t0))              # index of time closest to t0
            T_at_t0 = T_cp + F_total_B[idx_closest]
            P = P_temp_dependent(T_at_t0, h_sample_b, t_pi_B)
            F_total_B[mask] += P * (np.exp(-delta_t / t_th) - np.exp(-9 * delta_t / t_th))


    idx_meas = np.argmin(np.abs(t - t_meas))             
    Theta_A_meas = F_total_A[idx_meas]
    Theta_B_meas = F_total_B[idx_meas]

    Temp_A_list.append(Theta_A_meas)
    Temp_B_list.append(Theta_B_meas)


Temp_Siv_A = np.array(Temp_A_list) + T_a_array 
Temp_Siv_B = np.array(Temp_B_list) + T_b_array 


T_ratio_mod_list = []

for i, N in enumerate(N_values):
    # Load the CSV
    filename = f"N={N}_results_parallel.csv"
    df = pd.read_csv(filename)

    # Mask for eps = 0, f = 0 (central point)
    mask_center = (df['eps'] == 0) & (df['f'] == 0)

    # Extract T2_B central value
    T2_B = df.loc[mask_center, 't2_ideal'].values[0]

    # Extract full arrays
    T2_A = df["t2_grape"].to_numpy()
    eps_vals = df['eps'].to_numpy()
    f_vals = df['f'].to_numpy()

    # Get simulated temperatures (already computed)
    Temp_sim_A = Temp_Siv_A[i]
    Temp_sim_B = Temp_Siv_B[i]

    # Apply decoherence correction
    T2_A_mod = T2_A / (1 + T2_A * (Temp_sim_A - 0.1) * 3e6)
    T2_B_mod = T2_B / (1 + T2_B * (Temp_sim_B - 0.1) * 3e6)



    # Compute ratio
    T2_ratio = T2_A_mod / T2_B_mod

        # Create grid
    eps_unique = np.sort(np.unique(eps_vals))
    f_unique = np.sort(np.unique(f_vals))

    # Prepare 2D array for heatmap
    heatmap = np.empty((len(f_unique), len(eps_unique)))
    heatmap[:] = np.nan  # initialize with NaNs for missing data points

    
    # # Choose fixed min and max values for colorbar scale
    # vmin = 0.1  # or np.nanmin(T2_ratio) if you want data-driven
    # vmax = 20  # or np.nanmax(T2_ratio)

        # Choose fixed min and max values for colorbar scale
    vmin = 0.1  # or np.nanmin(T2_ratio) if you want data-driven
    vmax = 15  # or np.nanmax(T2_ratio)


    # Fill heatmap with T2_ratio values
    for x, y, val in zip(eps_vals, f_vals, T2_ratio):
        ix = np.where(eps_unique == x)[0][0]
        iy = np.where(f_unique == y)[0][0]
        heatmap[iy, ix] = val  # note: rows = f, cols = eps

    plt.figure(figsize=(8,6))
    im = plt.imshow(heatmap, 
                    extent=[eps_unique[0], eps_unique[-1], f_unique[0], f_unique[-1]],
                    origin='lower',
                    aspect='auto',
                    cmap='plasma',
                    #vmin=vmin, 
                    #vmax=vmax
                    norm=LogNorm(vmin=vmin, vmax=vmax)
                    )
    

    fmt = {1: '1', 2: '2', 4: '4', 8: '8', 16: '16'}
    #fmt = {16: '16', 20: '20'}

    
    # Overlay contour at T2 ratio = 1
    # Overlay contour at T2 ratio = 1, 2, 4, 8, 16
    X, Y = np.meshgrid(eps_unique, f_unique)
    levels = [1, 2, 4, 8, 16]
    #levels = [16, 20]
    contour = plt.contour(
        X, Y, heatmap, levels=levels, colors='black', linewidths=1.8
    )
    # Make contour lines dotted
    for c in contour.collections:
        c.set_linestyle('dotted')  # or 'dashed' / (0, (5, 5)) for custom

    # Add contour labels
    plt.clabel(contour, fmt={lvl: str(lvl) for lvl in levels}, inline=True, fontsize=22, colors='black')

    # Correct colorbar setup
    cbar = plt.colorbar(im)
    cbar.set_label(r'$T_{2\mathcal{A}} / T_{2\mathcal{B}}$', fontsize=28)
    cbar.ax.tick_params(labelsize=28)

    # Axis ticks and labels
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.xlabel(r'$\epsilon$', fontsize=32)
    plt.ylabel(r'$f$', fontsize=32)

    # Save plot
    plt.tight_layout()
    plt.savefig(f"T2_ratio_heatmap_N={N}.pdf")
    plt.close()




