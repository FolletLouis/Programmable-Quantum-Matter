import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import plasma

# Fixed parameters
rise_rate = 5.2e-3  # fractional rise in temperature per uW
h_a = 26.34         # uW
h_b = 65            # uW
t_pi_a = 150e-9     # s
t_pi_b = 10e-9      # s
m = 121             # fixed m value

# Ranges
N_values = np.array([2, 4, 8, 16])   # N values to plot
t_meas_values = [2e-4, 8e-4, 2e-3]                 # measurement times

T_cp = 100  # mK

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))
colors = plasma(np.linspace(0, 1, len(t_meas_values)*2+2))

for idx, t_meas in enumerate(t_meas_values):
    # Compute duty cycles
    dc_a_array = (t_pi_a * N_values) / t_meas
    dc_b_array = (t_pi_b * N_values * m) / t_meas

    # Compute effective heating powers
    h_a_eff_array = dc_a_array * h_a       # uW
    h_b_eff_array = dc_b_array * h_b       # uW

    # Compute fractional temperature rise
    frac_t_a_array = h_a_eff_array * rise_rate
    frac_t_b_array = h_b_eff_array * rise_rate

    # Absolute temperatures
    T_a_array = (1 + frac_t_a_array) * T_cp
    T_b_array = (1 + frac_t_b_array) * T_cp

    # Plot for t_meas
    ax1.plot(N_values, T_a_array, 'o--', color=colors[idx*2],
             label=rf'$\mathcal{{A}}$, $t_{{dds}}={t_meas*1e3:.1f}$ ms',
             linewidth=2, markersize=8)
    ax1.plot(N_values, T_b_array, 's-', color=colors[idx*2+1],
             label=rf'$\mathcal{{B}}$, $t_{{dds}}={t_meas*1e3:.1f}$ ms',
             linewidth=2, markersize=8)

# Customize axis
ax1.set_xlabel(r'$N$ (number of pulses)', fontsize=22)
ax1.set_ylabel(r'$\overline{\Theta}_{CP}$ (mK)', fontsize=22, color='black')
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_xticks(N_values)
ax1.set_xticklabels(N_values, fontsize=18)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xscale('log', base=2)

# Add legend
ax1.legend(fontsize=18, loc='upper left', frameon=False)

# Adjust layout
fig.set_constrained_layout(True)
plt.savefig('T_CP.pdf')
plt.show()
