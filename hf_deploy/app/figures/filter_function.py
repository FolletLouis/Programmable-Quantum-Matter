"""
Fig. 3: Filter function F_ij/ω² vs ωτ.
Source: Filter_Function_I.ipynb in Dynamical Decoupling - Filter Function
Interactive: Compare sequences A (SAFE-GRAPE) vs B (bang-bang) for varying N and f.
Requires: filter_functions, qutip (pip install filter-functions qutip)
"""
import numpy as np
import matplotlib.pyplot as plt

# SAFE-GRAPE pulse parameters from notebook
PHI = np.array([
    0.15040507, 0.22662143, 0.81338164, 1.60882984, 2.36321953, 2.9309562,
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
    -1.51979327, -0.32714796, 1.35224799, 2.96877042,
])
T_HISTORY = np.array([
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
    1.29965087e-09, 1.26135342e-09, 1.24133773e-09, 1.25059769e-09,
])


def _check_deps():
    """Raise ImportError if filter_functions or qutip not available."""
    try:
        import filter_functions as ff
        import qutip as qt
        return ff, qt
    except ImportError as e:
        raise ImportError(
            "Filter function plot requires filter-functions and qutip. "
            "Install with: pip install filter-functions qutip"
        ) from e


def _generate_pulse_sequence(ff, qt, f, eps, tau, N, mode):
    """Generate pulse sequence and filter function. From Filter_Function_I.ipynb."""
    OMEGA = 0.2e9
    epsilon_0 = 10
    tau_cycle = tau / N
    X, Y, Z = qt.sigmax(), qt.sigmay(), qt.sigmaz()

    if mode == "Ideal":
        tau_pi = 10e-9
        if tau_cycle <= tau_pi:
            raise ValueError("tau_cycle must be greater than tau_pi for Ideal mode")
        J = (1 + eps) * np.array([0, np.pi / tau_pi, 0])
        dBz = [f * OMEGA] * 3
        H_c = [[X / 2, J], [Z / 2, dBz]]
        H_n = [[X / 2, J / epsilon_0], [Z / 2, [1] * 3]]
        dt = [(tau_cycle - tau_pi) / 2, tau_pi, (tau_cycle - tau_pi) / 2]
        SE = ff.PulseSequence(H_c, H_n, dt)
        SE_n = ff.concatenate_periodic(SE, N)
        omega_list = ff.util.get_sample_frequencies(SE_n, n_samples=25000, spacing="log")
        FF = SE_n.get_filter_function(omega_list)
        return FF[1][1], omega_list * tau

    elif mode == "Grape":
        tau_pi1 = np.sum(T_HISTORY)
        if tau_cycle <= tau_pi1:
            raise ValueError("tau_cycle must be greater than tau_pi1 for Grape mode")
        dt1 = [(tau_cycle - tau_pi1) / 2]
        Cx, Cy, Nx, Ny = [0.0], [0.0], [0.0], [0.0]
        Nz = [1] * (len(PHI) + 2)
        for t in T_HISTORY:
            dt1.append(t)
        dt1.append((tau_cycle - tau_pi1) / 2)
        for cx in OMEGA * (1 + eps) * np.cos(PHI):
            Cx.append(cx)
            Nx.append(cx / epsilon_0)
        Cx.append(0)
        Nx.append(0)
        for cy in OMEGA * (1 + eps) * np.sin(PHI):
            Cy.append(cy)
            Ny.append(cy / epsilon_0)
        Cy.append(0)
        Ny.append(0)
        Cz = f * OMEGA * np.ones(len(PHI) + 2)
        H_c1 = [[X / 2, Cx], [Y / 2, Cy], [Z / 2, Cz]]
        H_n1 = [[X / 2, Nx], [Y / 2, Ny], [Z / 2, Nz]]
        SE1 = ff.PulseSequence(H_c1, H_n1, dt1)
        SE1_n = ff.concatenate_periodic(SE1, N)
        omega_list = ff.util.get_sample_frequencies(SE1_n, n_samples=25000, spacing="log")
        FF = SE1_n.get_filter_function(omega_list)
        return FF[2][2], omega_list * tau
    else:
        raise ValueError("mode must be 'Ideal' or 'Grape'")


def fig3_filter_function(
    N: int = 2,
    f: float = 0.0,
    eps: float = 0.0,
    tau: float = 5e-3,
) -> plt.Figure:
    """
    Interactive Fig. 3: F_ij/ω² vs ωτ for sequences A (Grape) and B (Ideal).
    """
    ff, qt = _check_deps()
    cmap = plt.colormaps["plasma"]
    color_a = cmap(0.2)
    color_b = cmap(0.8)

    FF_A, omegatau_A = _generate_pulse_sequence(ff, qt, f, eps, tau, N, "Grape")
    FF_B, omegatau_B = _generate_pulse_sequence(ff, qt, f, eps, tau, N, "Ideal")

    n_pts = min(25000, len(FF_A), len(FF_B))
    FF_A = np.real(np.asarray(FF_A[:n_pts]))
    FF_B = np.real(np.asarray(FF_B[:n_pts]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(omegatau_A[:n_pts], FF_A, color=color_a, label=rf"$\mathcal{{A}}_{{{N}}}$")
    ax.plot(omegatau_B[:n_pts], FF_B, "--", color=color_b, label=rf"$\mathcal{{B}}_{{{N}}}$")
    ax.set_xlabel(r"$\omega\tau$")
    ax.set_ylabel(r"$F_{ij}/\omega^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(rf"Filter function — $N={N}$, $f={f:.2f}$, $\epsilon={eps:.2f}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
