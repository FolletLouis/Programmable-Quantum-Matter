import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import plasma
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.ticker import LogLocator

colors = plasma(np.linspace(0, 0.7, 2))
e_th = 0.5

N = [2,4,8,16]
# t_meas = 2e-4 s, t_cmpl = 1e-6 s
a_num1 = np.array([10403, 9379, 8207, 5998]) 
a_e_avg1 = np.array([0.1862942603171783, 0.19019864458852706, 0.22976180602164323, 0.2963147865540428])


# t_meas = 2e-4 s, t_cmpl = 1e-5 s
a_num2 = [10231, 9221, 8029, 5857] 
a_e_avg2 = [0.18986770883242557, 0.19106734039838802, 0.23047830018185766, 0.29912289777987533]


# t_meas = 8e-4 s, t_cmpl = 1e-6 s
a_num3 = [0, 573, 930, 441] 
a_e_avg3 = [1.0, 0.3930422557816903, 0.35708101009412, 0.3936174722932882]


# t_meas = 8e-4 s, t_cmpl = 1e-5 s
a_num4 = [0, 552, 926, 447] 
a_e_avg4 = [1.0, 0.3937116264805626, 0.356472854011253, 0.39186830473476886]


############################################################

# t_meas = 2e-4 s, t_cmpl = 1e-6 s
b_num1 = np.array([14641, 0, 0, 0]) 
b_e_avg1 = np.array([0.2916677341713093, 0.9359266, 1, 1])


# t_meas = 2e-4 s, t_cmpl = 1e-5 s
b_num2 = [14641, 0, 0, 0] 
b_e_avg2 = [0.2970211091994934, 0.937134, 1, 1]


# t_meas = 8e-4 s, t_cmpl = 1e-6 s
b_num3 = [0, 0, 0, 0] 
b_e_avg3 = [0.9997, 0.9999945, 1, 1]


# t_meas = 8e-4 s, t_cmpl = 1e-5 s
b_num4 = [0, 0, 0, 0] 
b_e_avg4 = [0.9997, 0.999989, 1, 1]


fig, ax = plt.subplots(figsize=(8,6), dpi=120)

ax.plot(N, a_e_avg4, 'o-', color=colors[0], label=r'$\mathcal{A}$', linewidth=1.8)
ax.plot(N, b_e_avg4, 'o--', color=colors[0], label=r'$\mathcal{B}$', linewidth=1.8)


# Shade region above threshold
ax.axhspan(e_th, 1.0, facecolor='none', hatch='//', edgecolor=colors[1], alpha=0.6)

# Threshold reference line
ax.axhline(y=e_th, color=colors[1], linestyle="--", linewidth=1.8, label = r'$\epsilon_{th}$')

ax.set_xscale("log", base=2)
ax.set_xlabel("N", fontsize=28)
ax.set_ylabel(r"$\epsilon_{eff}$", fontsize=28)

# Make tick labels big too
ax.tick_params(axis='both', which='major', labelsize=23)

ax.grid(True)
ax.legend(fontsize=23, frameon=False)

plt.tight_layout()
plt.savefig("e_eff_vs_N_IV.pdf")
plt.show()