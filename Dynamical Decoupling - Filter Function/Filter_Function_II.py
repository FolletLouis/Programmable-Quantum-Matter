import filter_functions as ff
from filter_functions import analytic, plotting
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from math import pi
import numpy as np
import csv

import scipy.integrate
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor


# Pauli matrices
X, Y, Z = qt.sigmax(), qt.sigmay(), qt.sigmaz()

phi = np.array([0.15040507, 0.22662143, 0.81338164, 1.60882984, 2.36321953, 2.9309562,
       3.28516708, 3.45474715, 3.45526928, 3.23345478, 2.61349612, 1.32113029,
       6.07523601, 5.57997002, 5.30330316, 5.16714173, 5.11696132, 5.10504146,
       5.09281771, 5.06053417, 5.01310673, 4.97551, 4.97366974, 5.01145251,
       5.06880815, 5.12563276, 5.18132239, 5.24721174, 5.32733826, 5.41424006,
       5.49967307, 5.58035189, 5.65086264, 5.69273024, 5.66329514, 5.47818519,
       2.18587196, 1.69954578, 1.63502883, 1.65029134, 1.58102867, 1.42275827,
       1.26989491, 1.22646316, 1.32296113, 1.52009356, 1.81069506, 2.16567911,
       -1.02393239, -1.19819281, -0.98608233, -0.86421354, -0.83115883, -0.82210699,
       -0.81340814, -0.80170519, -0.77641705, -0.72212651, -0.64007123, -0.57036507,
       -0.58814509, -0.75584938, -1.05454523, -1.35897787, -1.50255204, -1.36784821,
       -0.94512912, -0.38398199, 0.0629544, 0.23886577, 0.1616152, -0.04556237,
       -0.22770354, -0.25495998, -0.07294665, 0.2563205, 4.3174622, 3.75656933,
       3.39290744, 3.13520017, 2.94016075, 2.7974009, 2.70460944, 2.65801828,
       2.65447597, 2.69459055, 2.78042136, 2.90940093, 3.07128212, 3.252744,
       3.44621644, 3.65657649, 3.90264854, 4.21123338, 4.60144671, 5.06268089,
       -1.51979327, -0.32714796, 1.35224799, 2.96877042])

t_history = np.array([
    1.24462306e-09, 1.24860635e-09, 1.24536306e-09, 1.24000370e-09,
    1.24010816e-09, 1.24776900e-09, 1.26046912e-09, 1.27354216e-09,
    1.28205382e-09, 1.28220453e-09, 1.27306604e-09, 1.26350346e-09,
    1.40005554e-09, 1.39692284e-09, 1.39635931e-09, 1.39761874e-09,
    1.40002240e-09, 1.40275374e-09, 1.40566233e-09, 1.40915859e-09,
    1.41342158e-09, 1.41760949e-09, 1.41975079e-09, 1.41807213e-09,
    1.41299337e-09, 1.40739326e-09, 1.40449323e-09, 1.40546704e-09,
    1.40900757e-09, 1.41300542e-09, 1.41619064e-09, 1.41815891e-09,
    1.41865629e-09, 1.41736051e-09, 1.41421786e-09, 1.41023353e-09,
    1.60217733e-09, 1.64179084e-09, 1.66173643e-09, 1.66907469e-09,
    1.67495120e-09, 1.67290026e-09, 1.65567694e-09, 1.63160390e-09,
    1.61176080e-09, 1.59553431e-09, 1.58101988e-09, 1.57288918e-09,
    1.46987405e-09, 1.50028073e-09, 1.49871533e-09, 1.49087135e-09,
    1.48363993e-09, 1.47546980e-09, 1.46626969e-09, 1.46103343e-09,
    1.46342539e-09, 1.47148247e-09, 1.48039815e-09, 1.48727159e-09,
    1.49313602e-09, 1.50222128e-09, 1.51869863e-09, 1.53933393e-09,
    1.54993905e-09, 1.53949747e-09, 1.51794899e-09, 1.50585363e-09,
    1.50804167e-09, 1.51307862e-09, 1.51369933e-09, 1.51343242e-09,
    1.51736565e-09, 1.52494062e-09, 1.53001219e-09, 1.52193585e-09,
    1.29389970e-09, 1.30191641e-09, 1.30455682e-09, 1.30392924e-09,
    1.30273819e-09, 1.30222796e-09, 1.30226802e-09, 1.30212021e-09,
    1.30135639e-09, 1.30035961e-09, 1.29994920e-09, 1.30062258e-09,
    1.30233740e-09, 1.30479273e-09, 1.30756428e-09, 1.31015797e-09,
    1.31245075e-09, 1.31539167e-09, 1.32134776e-09, 1.33348913e-09,
    1.29965087e-09, 1.26135342e-09, 1.24133773e-09, 1.25059769e-09
])



def generate_pulse_sequence(f, eps, tau, N, mode):
    """
    Generates a PulseSequence for Spin Echo (SE) or Arbitrary Echo (SE1).

    Parameters:
        epsilon (float): Sensitivity parameter.
        f (float): Frequency scaling factor.
        tau (float): Total sequence duration.
        N (int): Number of concatenations
        mode (str): "Ideal" for Spin Echo, "Grape" for Arbitrary Echo.

    Returns:
        PulseSequence: The generated sequence.
    """

    # Define constants
    epsilon_0 = 10  # Sensitivity normalization factor
    OMEGA = 0.2e9  # Default Omega
    tau_cycle = tau/N

    if mode == "FID":
        dt = [tau_cycle]
        H_c = [
            [X/2, [0]],
            [Z/2, [1]]
            ]
            # Noise Hamiltonian
        H_n = [
            [X/2, [1]],
            [Z/2, [1]]
            ]

        FID = ff.PulseSequence(H_c, H_n, dt)
        omega_list = ff.util.get_sample_frequencies(FID, n_samples=50000, spacing='log')
        FF = FID.get_filter_function(omega_list)

        return [FID, FF[1][1], omega_list*tau]

    if mode == "Ideal":
        # Pi pulse duration for simple square pulse
        tau_pi = 10e-9
        if (tau_cycle > tau_pi):

          # Control Hamiltonian
          J = (1 + eps)*np.array([0, np.pi / tau_pi, 0])
          dBz = [f * OMEGA] * 3

          H_c = [
              [X/2, J],
              [Z/2, dBz]
          ]

          # Noise Hamiltonian
          H_n = [
              [X/2, J / epsilon_0],
              [Z/2, [1] * 3]  # B-field
          ]

          # Time steps
          dt = [(tau_cycle - tau_pi) / 2, tau_pi, (tau_cycle - tau_pi) / 2]

          SE = ff.PulseSequence(H_c, H_n, dt)
          SE_n = ff.concatenate_periodic(SE, N)

          omega_list = ff.util.get_sample_frequencies(SE_n, n_samples=50000, spacing='log')
          FF = SE_n.get_filter_function(omega_list)

          return [SE_n, FF[1][1], omega_list*tau]

        else:
          raise ValueError("tau_cycle must be greater than tau_pi.")

    elif mode == "Grape":
        # Assume 't_history' and 'phi' are pre-defined externally
        global t_history, phi

        # Pi-pulse duration for GRAPE pulse
        tau_pi1 = np.sum(t_history)

        if (tau_cycle > tau_pi1):

          # Time steps
          dt1 = [(tau_cycle - tau_pi1) / 2]

          Cx = [0.]
          Cy = [0.]
          Nx = [0.]
          Ny = [0.]
          Nz = [1] * (len(phi) + 2)

          # Construct control pulses
          for t in t_history:
              dt1.append(t)
          dt1.append((tau_cycle - tau_pi1) / 2)

          for cx in OMEGA * (1 + eps) * np.cos(phi):
              Cx.append(cx)
              Nx.append(cx / epsilon_0)

          Cx.append(0)
          Nx.append(0)

          for cy in OMEGA * (1 + eps) * np.sin(phi):
              Cy.append(cy)
              Ny.append(cy / epsilon_0)

          Cy.append(0)
          Ny.append(0)

          Cz = f * OMEGA * np.ones(len(phi) + 2)

          # Define Control Hamiltonian
          H_c1 = [
              [X/2, Cx],
              [Y/2, Cy],
              [Z/2, Cz]
          ]

          # Define Noise Hamiltonian
          H_n1 = [
              [X/2, Nx],
              [Y/2, Ny],
              [Z/2, Nz]  # B-field
          ]

          SE1 = ff.PulseSequence(H_c1, H_n1, dt1)
          SE1_n = ff.concatenate_periodic(SE1, N)

          omega_list = ff.util.get_sample_frequencies(SE1_n, n_samples=50000, spacing='log')
          FF = SE1_n.get_filter_function(omega_list)

          return [SE1_n, FF[2][2], omega_list*tau]


        else:
          raise ValueError("tau_cycle must be greater than tau_pi1.")

    else:
        raise ValueError("Invalid mode. Choose 'SE' or 'SE1'.")


def func(x, m, c):
    return (m*x + c)

# Define the noise spectrum S(omega) (example: 1/f noise with cutoff)
def S_w(f0, omega, omega_0, a, b, c):
    return f0 / (a + b*np.power(omega/omega_0, c))

def S_exp(omega, A, omega_0, B, omega_1):
    return (A * np.exp(- (omega/omega_0)**2) + B * np.exp(- (omega/omega_1)**2))

def pulse_integrator(f, eps, tau, N, mode, na, nb, nc, omega_0):
    P = generate_pulse_sequence(f, eps, tau, N, mode)
    FF = np.array(P[1])
    omegatau_list = np.array(P[2])
    omega_list = 1/tau*omegatau_list

    FF1 = FF/(tau**2)

    # Noise = S_w(1, omegatau_list/tau, omega_0, na, nb, nc)
    Noise = S_exp(omegatau_list/tau, 1e6, 1.8e3, 1e9, 50)
    integrand = tau * Noise * FF1
    integral_value = scipy.integrate.simpson(integrand, omegatau_list)
    return integral_value

def pulse_T2(f, eps, tau1, tau2, N, mode, na, nb, nc, omega_0, plotting = "on"):
    integral_values = []
    tau_values = np.linspace(tau1, tau2, 20)

    for t in tau_values:
        integral_value = pulse_integrator(f, eps, t, N, mode, na, nb, nc, omega_0)
        integral_values.append(integral_value)

    popt, pcov = curve_fit(func, np.log(tau_values), np.log(integral_values))
    alpha = popt[0]
    T2 = np.exp(-popt[1]/popt[0])

    if plotting == "on":
        plt.figure(figsize=(8, 5))
        plt.plot(np.log(tau_values), np.log(integral_values), marker='o', linestyle='-')
        plt.plot(np.log(tau_values), func(np.log(tau_values), *popt), 'r--', label='Fit for $T_2^*$')
        #plt.xlabel(r'$\text{ln}(\tau)$')
        #plt.ylabel(r'$\text{ln}(\chi)$')
        plt.title('Integral of Filter Function vs. $\tau$')
        plt.grid(True)
        plt.show()

    return [integral_values, tau_values, T2, alpha]



# --- Core Parallel Computation Function ---
def compute_results(args):
    f, eps, Nt = args
    result_grape = pulse_T2(f, eps, 2e-4, 5e-3, Nt, "Grape", 0, 1, 1, 10, "off")
    result_ideal = pulse_T2(f, eps, 2e-4, 5e-3, Nt, "Ideal", 0, 1, 1, 10, "off")
    t2_grape = result_grape[2]
    alpha_grape = result_grape[3]
    t2_ideal = result_ideal[2]
    alpha_ideal = result_ideal[3]
    t2_ratio = t2_grape / t2_ideal
    return [eps, f, t2_grape, t2_ideal, t2_ratio, alpha_grape, alpha_ideal]

# --- Run Computation ---
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional but recommended on Windows

    sweep_range = np.linspace(-0.5, 0.5, 11)
    f_values = sweep_range
    eps_values = sweep_range
    Nt = 16
    output_file = "N=16_results_parallel.csv"

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "eps", "f", "t2_grape", "t2_ideal", "t2_ratio", "alpha_grape", "alpha_ideal"])

        iter_counter = 0

        for eps in eps_values:
            with ProcessPoolExecutor() as executor:
                args_list = [(f, eps, Nt) for f in f_values]
                results = list(executor.map(compute_results, args_list))

            for row in results:
                iter_counter += 1
                writer.writerow([iter_counter] + [f"{x:.3e}" if isinstance(x, float) else x for x in row])
                print(iter_counter)









