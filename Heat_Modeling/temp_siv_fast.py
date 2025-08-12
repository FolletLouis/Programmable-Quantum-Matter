import numpy as np
import matplotlib.pyplot as plt
from heat_capacity import *
from matplotlib.cm import plasma

# Parameters
t_pi_A = 150e-9      # pulse duration for A
t_pi_B = 10e-9       # pulse duration for B
t_meas = 2e-4        # total measurement time
m = 121             # multiplier for B sequence
t_th = 70e-6         # thermalization timescale
T_cp = 100e-3        # baseline temp (K)


gamma_mech = 1e-3
# Time array
t = np.linspace(0, 2*t_meas, 100000)

# N values to test
N_values = [2, 4, 8, 16]


# h_sample 
h_sample_a = gamma_mech * 0.084e-6  # W
h_sample_b = gamma_mech * 0.207e-6 #W

# Compute P(T) from formula
def P_temp_dependent(T, h_sample, t_pi):
    return ((9/8) * h_sample * t_pi) / (np.power(9,-1/8) * Cv(T))


colors = plasma(np.linspace(0, 0.8, len(N_values)))
# Create plot
plt.figure(figsize=(10, 6))
i = 0

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

    F_total_A += T_cp
    F_total_B += T_cp
    # Plot
    plt.plot(t * 1e3, F_total_A*1e3 , '-', color = colors[i], label=rf'$\mathcal{{A}}_{{{N}}}$', linewidth=1.8)
    #plt.plot(t * 1e3, F_total_B*1e3, '--', color = colors[i], label=rf'$\mathcal{{B}}, N={N}$', linewidth=1.8)
    i +=1

# Labels & Legend
#plt.yscale('log')
plt.axvline(x=t_meas*1e3, color='k', linestyle=':', linewidth=1.8, label=r'$t_{dds}$')
plt.tick_params(axis='both', which='major', labelsize=23)  # This sets tick label font size
plt.xlabel('Time $t$ (ms)', fontsize=28)
plt.ylabel(r'$\Theta_{\mathrm{SiV}}$ (mK)', fontsize=28)
plt.legend(fontsize=22)
plt.grid(True)
plt.tight_layout()

plt.show()
