import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from T2T1 import *

# Set global font to Arial
mpl.rcParams.update({
    'font.size': 27,
    'font.family': 'Arial',
    'pdf.fonttype': 42
})

plt.figure(figsize=(6, 6))

# Parameters
Ns = [16]
alpha_ent = 1e-4
eta = 0.01
t_cmpl = 1e-6 #  np.logspace(-5, -3, 5)
t_meas = 2e-4 # s


e_th = 0.5
T1_orig = 10 #sec

colors = plt.cm.plasma(np.linspace(0, 0.7, 6))

# Load Fidelity once
F_sim = pd.read_csv("Fid_N=2.csv")

for idx, N in enumerate(Ns):

    e_avg = 0
    Nt = N
    file_prefix = f"N={N}_results_parallel.csv"
    df = pd.read_csv(file_prefix)

    # Extract data
    T2 = df['t2_grape'].values
    scale = df['alpha_grape'].values
    eps = df['eps'].values
    f = df['f'].values
    F_data = (F_sim['Fidelity Grape'].values)

    F_A = np.array(F_data)
    F_B = np.array(F_data)
    T2_A = np.array(T2)
    T2_B = np.array(T2)
    scl_A = np.array(scale)
    scl_B = np.array(scale)
    eps_A = np.array(eps)
    eps_B = np.array(eps)
    f_A = np.array(f)
    f_B = np.array(f)

    
    count = 0
    good_links = 0
    labeled = False
    for i in range(len(df)):
        for j in range(len(df)):
            if np.abs(f_A[i]) < 0.6 and np.abs(eps_A[i]) < 0.6 and \
               np.abs(f_B[j]) < 0.6 and np.abs(eps_B[j]) < 0.6:
                
                Fdds = F_A[i]*F_B[j]


                count +=1
                print(count)

                if (Fdds < 0.5):
                    continue
                
                else:
                    e_tot = epsilon_jk_avg(t_cmpl, t_meas, N, "A" , T1_orig, T2_A[i], T2_B[j], scl_A[i], scl_B[j], Fdds)

                    if (e_tot < e_th):
                        e_avg += e_tot
                        good_links += 1

    if (good_links > 0):
        e_avg = e_avg/good_links

    print(N, e_avg, good_links)
