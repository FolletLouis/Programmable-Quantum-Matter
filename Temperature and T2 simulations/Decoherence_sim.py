import numpy as np
import matplotlib.pyplot as plt
from heat_capacity import *
from matplotlib.cm import plasma
import pandas as pd
from matplotlib.colors import to_hex



# Parameters
t_pi_A = 150e-9      # pulse duration for A
t_pi_B = 10e-9       # pulse duration for B
t_meas = 2e-4        # total measurement time
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


colors = plasma(np.linspace(0, 0.7, 2))

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


T2A_mod_list = []
T2B_mod_list = []

for i, N in enumerate(N_values):
    # Load the CSV
    filename = f"N={N}_results_parallel.csv"
    df = pd.read_csv(filename)

    # Mask for eps = 0, f = 0
    mask = (df['eps'] == 0) & (df['f'] == 0)

    # Extract raw T2 values
    T2_A = df.loc[mask, 't2_grape'].values[0]
    T2_B = df.loc[mask, 't2_ideal'].values[0]

    # Get simulated temperatures (already computed)
    Temp_sim_A = Temp_Siv_A[i]
    Temp_sim_B = Temp_Siv_B[i]

    # Apply decoherence correction
    T2_A_mod = T2_A / (1 + T2_A * (Temp_sim_A - 0.1) * 3e6)
    T2_B_mod = T2_B / (1 + T2_B * (Temp_sim_B - 0.1) * 3e6)

    # Store for plotting
    T2A_mod_list.append(T2_A_mod)
    T2B_mod_list.append(T2_B_mod)

# Convert to arrays
T2A_mod_array = np.array(T2A_mod_list)
T2B_mod_array = np.array(T2B_mod_list)

fig, ax1 = plt.subplots(figsize=(7,6))

color1 = to_hex(colors[0])
color2 = to_hex(colors[1])


# Plot on left y-axis
ax1.plot(N_values, T2A_mod_array * 1e6, 'o-', color=colors[0], label=r'$T_{2\mathcal{A}}$', linewidth=1.8)
ax1.plot(N_values, T2B_mod_array * 1e6, 'o--', color=colors[0], label=r'$T_{2\mathcal{B}}$', linewidth=1.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.yaxis.label.set_color(color1)
ax1.set_xlabel("N", fontsize=28)
ax1.set_ylabel(r"$T_2$ ($\mu$s)", fontsize=28)
ax1.set_xscale('log', base=2)

ax1.tick_params(axis='both', which='major', labelsize=23)
ax1.grid(True)

# Create twin axis for right y-axis
#ax2 = ax1.twinx()

# # Plot on right y-axis
# ax1.plot(N_values, 1e3*Temp_Siv_A, 's-', color=colors[1], label=r'$\Theta_{\mathcal{A}}$', linewidth=1.8)
# ax1.plot(N_values, 1e3*Temp_Siv_B, 's--', color=colors[1], label=r'$\Theta_{\mathcal{B}}$', linewidth=1.8)
# ax1.tick_params(axis='y', labelcolor=color2)
# ax1.yaxis.label.set_color(color2)
# ax1.set_ylabel(r"$\Theta_{SiV}$ (mK)", fontsize=28)
# ax1.tick_params(axis='y', labelsize=23)

# Horizontal line at measurement time
ax1.axhline(y=t_meas*1e6, color=colors[1], linestyle='--', linewidth=1.5, label=r'$t_{\mathrm{dds}}$')
ax1.fill_between(N_values, 0, t_meas*1e6, facecolor='none', hatch='//', edgecolor=colors[1], alpha=0.5)
ax1.legend(fontsize=20, frameon=False)
# Combine legends from both axes
#lines_1, labels_1 = ax1.get_legend_handles_labels()
#lines_2, labels_2 = ax2.get_legend_handles_labels()
#ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=20, frameon=False)


plt.tight_layout()
#plt.savefig("T2_vs_N_tmeas_2e-4.pdf")
plt.show()