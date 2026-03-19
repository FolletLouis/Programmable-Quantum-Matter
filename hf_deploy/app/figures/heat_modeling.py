"""
Heat modeling: Fig. 11 (dilution fridge), Fig. 12 (SiV fast temp).
EXACT-PAPER REPRODUCTION from dil_fridge.py, temp_siv_fast.py.
"""
import numpy as np
import matplotlib.pyplot as plt

from ..heat_capacity import Cv

# From dil_fridge.py
rise_rate = 5.2e-3
h_a, h_b = 26.34, 65
t_pi_a, t_pi_b = 150e-9, 10e-9
m = 121
T_cp = 100  # mK
N_values = np.array([2, 4, 8, 16])
t_meas_values = [2e-4, 8e-4, 2e-3]


def fig11_dil_fridge(t_meas_filter: list[float] | None = None) -> plt.Figure:
    """
    Fig. 11 (SI): Cold-plate temperature vs N for sequences A and B.
    Source: dil_fridge.py
    """
    use_vals = t_meas_filter if t_meas_filter is not None else t_meas_values
    colors = plt.cm.plasma(np.linspace(0, 1, len(t_meas_values) * 2 + 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, t_meas in enumerate(t_meas_values):
        if t_meas not in use_vals:
            continue
        dc_a = (t_pi_a * N_values) / t_meas
        dc_b = (t_pi_b * N_values * m) / t_meas
        T_a = (1 + dc_a * h_a * rise_rate) * T_cp
        T_b = (1 + dc_b * h_b * rise_rate) * T_cp
        ax.plot(N_values, T_a, "o--", color=colors[idx * 2],
                label=rf"$\mathcal{{A}}$, $t_{{dds}}={t_meas*1e3:.1f}$ ms", linewidth=2, markersize=8)
        ax.plot(N_values, T_b, "s-", color=colors[idx * 2 + 1],
                label=rf"$\mathcal{{B}}$, $t_{{dds}}={t_meas*1e3:.1f}$ ms", linewidth=2, markersize=8)
    ax.set_xlabel(r"$N$ (number of pulses)")
    ax.set_ylabel(r"$\overline{\Theta}_{CP}$ (mK)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def fig12_siv_fast(t_meas: float = 2e-4, sequence: str = "A") -> plt.Figure:
    """
    Fig. 12 (SI): SiV fast temperature rise vs time.
    Source: temp_siv_fast.py
    sequence: "A" or "B"
    """
    t_pi_A, t_pi_B = 150e-9, 10e-9
    t_th = 70e-6
    T_cp = 100e-3  # K
    gamma_mech = 1e-3
    h_sample_a = gamma_mech * 0.084e-6
    h_sample_b = gamma_mech * 0.207e-6

    def P_temp_dependent(T, h_sample, t_pi):
        beta = np.power(9, -1 / 8)
        return (9 / 8) * h_sample * t_pi / (beta * Cv(T))

    t = np.linspace(0, 2 * t_meas, 100000)
    N_values = [2, 4, 8, 16]
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(N_values)))

    use_A = sequence.upper() == "A"
    h_sample = h_sample_a if use_A else h_sample_b
    t_pi = t_pi_A if use_A else t_pi_B
    seq_label = r"$\mathcal{A}$" if use_A else r"$\mathcal{B}$"

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, N in enumerate(N_values):
        F_total = np.zeros_like(t)
        if use_A:
            t0_list = [(2 * j + 1) * t_meas / (2 * N) for j in range(N)]
        else:
            t0_list = []
            for j in range(N):
                t_j = (2 * j + 1) * t_meas / (2 * N)
                for k in range(1, m + 1):
                    t0 = t_j + (k - 1) * t_pi_B
                    if t0 < t[-1]:
                        t0_list.append(t0)
        for t0 in t0_list:
            mask = t > t0
            delta_t = t[mask] - t0
            idx_closest = np.argmin(np.abs(t - t0))
            T_at_t0 = T_cp + F_total[idx_closest]
            P = P_temp_dependent(T_at_t0, h_sample, t_pi)
            F_total[mask] += P * (np.exp(-delta_t / t_th) - np.exp(-9 * delta_t / t_th))
        F_total += T_cp
        ax.plot(t * 1e3, F_total * 1e3, "-", color=colors[i], label=rf"{seq_label}$_{{{N}}}$", linewidth=1.8)
    ax.axvline(x=t_meas * 1e3, color="k", linestyle=":", linewidth=1.8, label=r"$t_{dds}$")
    ax.set_xlabel("Time $t$ (ms)")
    ax.set_ylabel(r"$\Theta_{\mathrm{SiV}}$ (mK)")
    ax.legend()
    ax.grid(True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig
