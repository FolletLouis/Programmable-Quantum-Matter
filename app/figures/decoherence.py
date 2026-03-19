"""
Temperature and T2 section: Fig. 4(a-f) and Fig. 4(g-j).
Uses precomputed CSV from Filter_Function_II.py.
FAST MODE: Load from CSV. FULL MODE: Would require running Filter_Function_II (expensive).
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from pathlib import Path

from ..paths import TEMPERATURE_T2, N_VALUES, csv_path
from ..heat_capacity import Cv


# Paper parameters (from Decoherence_sim.py, T2_heatmaps.py)
t_pi_A = 150e-9
t_pi_B = 10e-9
m = 121
t_th = 70e-6
T_cp = 100e-3
gamma_mech = 1e-3
h_sample_a = gamma_mech * 0.084e-6
h_sample_b = gamma_mech * 0.207e-6
rise_rate = 5.2e-3
h_a, h_b = 26.34, 65


def _compute_temp_evolution(t_meas: float, N_values: np.ndarray) -> tuple:
    """Compute SiV temperature evolution for sequences A and B."""
    t = np.linspace(0, 2 * t_meas, 10000)
    dc_a = (t_pi_A * N_values) / t_meas
    dc_b = (t_pi_B * N_values * m) / t_meas
    T_a_array = (1 + dc_a * h_a * rise_rate) * T_cp
    T_b_array = (1 + dc_b * h_b * rise_rate) * T_cp

    def P_temp(T, h_sample, t_pi):
        return ((9/8) * h_sample * t_pi) / (np.power(9, -1/8) * Cv(T))

    Temp_A_list, Temp_B_list = [], []
    for N in N_values:
        F_A, F_B = np.zeros_like(t), np.zeros_like(t)
        t0_As = [(2*j + 1) * t_meas / (2*N) for j in range(N)]
        for t0 in t0_As:
            if t0 >= t[-1]:
                continue
            mask = t > t0
            delta_t = t[mask] - t0
            idx = np.argmin(np.abs(t - t0))
            T_at = T_cp + F_A[idx]
            P = P_temp(T_at, h_sample_a, t_pi_A)
            F_A[mask] += P * (np.exp(-delta_t/t_th) - np.exp(-9*delta_t/t_th))
        for j in range(N):
            t_j = (2*j + 1) * t_meas / (2*N)
            for k in range(1, m + 1):
                t0 = t_j + (k - 1) * t_pi_B
                if t0 >= t[-1]:
                    continue
                mask = t > t0
                delta_t = t[mask] - t0
                idx = np.argmin(np.abs(t - t0))
                T_at = T_cp + F_B[idx]
                P = P_temp(T_at, h_sample_b, t_pi_B)
                F_B[mask] += P * (np.exp(-delta_t/t_th) - np.exp(-9*delta_t/t_th))
        idx_meas = np.argmin(np.abs(t - t_meas))
        Temp_A_list.append(F_A[idx_meas])
        Temp_B_list.append(F_B[idx_meas])

    Temp_Siv_A = np.array(Temp_A_list) + T_a_array
    Temp_Siv_B = np.array(Temp_B_list) + T_b_array
    return Temp_Siv_A, Temp_Siv_B


def fig4_t2_vs_N(t_meas: float = 2e-4) -> plt.Figure:
    """
    Fig. 4(a-f) simplified: T2 vs N for sequences A and B.
    EXACT-PAPER REPRODUCTION when CSV exists. CACHED from Filter_Function_II output.
    """
    N_arr = np.array(N_VALUES)
    Temp_Siv_A, Temp_Siv_B = _compute_temp_evolution(t_meas, N_arr)
    colors = plt.cm.plasma(np.linspace(0, 0.7, 2))
    T2A_list, T2B_list = [], []

    for i, N in enumerate(N_VALUES):
        path = csv_path(N)
        if not path.exists():
            # No CSV: show placeholder
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "Run Filter_Function_II.py to generate CSV files\n"
                    "(Temperature and T2 simulations folder)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig
        df = pd.read_csv(path)
        mask = (df["eps"] == 0) & (df["f"] == 0)
        masked = df.loc[mask]
        if len(masked) == 0:
            T2A_list.append(np.nan)
            T2B_list.append(np.nan)
            continue
        T2_A = masked["t2_grape"].values[0]
        T2_B = masked["t2_ideal"].values[0]
        T_sim_A, T_sim_B = Temp_Siv_A[i], Temp_Siv_B[i]
        T2_A_mod = T2_A / (1 + T2_A * (T_sim_A - 0.1) * 3e6)
        T2_B_mod = T2_B / (1 + T2_B * (T_sim_B - 0.1) * 3e6)
        T2A_list.append(T2_A_mod)
        T2B_list.append(T2_B_mod)

    T2A_arr = np.array(T2A_list) * 1e6
    T2B_arr = np.array(T2B_list) * 1e6
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(N_arr, T2A_arr, "o-", color=colors[0], label=r"$T_{2\mathcal{A}}$", linewidth=1.8)
    ax.plot(N_arr, T2B_arr, "o--", color=colors[0], label=r"$T_{2\mathcal{B}}$", linewidth=1.8)
    ax.axhline(y=t_meas * 1e6, color=colors[1], linestyle="--", linewidth=1.5, label=r"$t_{\mathrm{dds}}$")
    ax.fill_between(N_arr, 0, t_meas * 1e6, facecolor="none", hatch="//", edgecolor=colors[1], alpha=0.5)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$T_2$ ($\mu$s)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=12)
    ax.grid(True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def fig4_t2_heatmap(N: int = 4, t_meas: float = 2e-4) -> plt.Figure:
    """
    Fig. 4(g-j): T2_A / T2_B heatmap over (epsilon, f) grid.
    CACHED from Filter_Function_II CSV.
    """
    path = csv_path(N)
    if not path.exists():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "CSV not found. Run Filter_Function_II.py.", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    df = pd.read_csv(path)
    mask_center = (df["eps"] == 0) & (df["f"] == 0)
    center_vals = df.loc[mask_center, "t2_ideal"].values
    if len(center_vals) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, f"No (eps=0, f=0) row in CSV for N={N}. Check Filter_Function_II output.", ha="center", va="center", transform=ax.transAxes)
        return fig
    T2_B = center_vals[0]
    T2_A = df["t2_grape"].to_numpy()
    eps_vals = df["eps"].to_numpy()
    f_vals = df["f"].to_numpy()

    # Use same temp evolution as T2 vs N for consistency
    N_arr = np.array([N])
    Temp_Siv_A, Temp_Siv_B = _compute_temp_evolution(t_meas, N_arr)
    Temp_sim_A, Temp_sim_B = float(Temp_Siv_A[0]), float(Temp_Siv_B[0])
    T2_A_mod = T2_A / (1 + T2_A * (Temp_sim_A - 0.1) * 3e6)
    T2_B_mod = T2_B / (1 + T2_B * (Temp_sim_B - 0.1) * 3e6)
    T2_ratio = T2_A_mod / T2_B_mod

    eps_unique = np.sort(np.unique(eps_vals))
    f_unique = np.sort(np.unique(f_vals))
    if len(eps_unique) == 0 or len(f_unique) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, f"CSV for N={N} has no valid (eps, f) grid.", ha="center", va="center", transform=ax.transAxes)
        return fig
    heatmap = np.full((len(f_unique), len(eps_unique)), np.nan)
    for x, y, val in zip(eps_vals, f_vals, T2_ratio):
        ix_match = np.where(np.isclose(eps_unique, x))[0]
        iy_match = np.where(np.isclose(f_unique, y))[0]
        if len(ix_match) > 0 and len(iy_match) > 0:
            heatmap[iy_match[0], ix_match[0]] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, extent=[eps_unique[0], eps_unique[-1], f_unique[0], f_unique[-1]],
                   origin="lower", aspect="auto", cmap="plasma", norm=LogNorm(vmin=0.1, vmax=15))
    X, Y = np.meshgrid(eps_unique, f_unique)
    contour = ax.contour(X, Y, heatmap, levels=[1, 2, 4, 8, 16], colors="black", linewidths=1)
    ax.clabel(contour, inline=True, fontsize=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$T_{2\mathcal{A}} / T_{2\mathcal{B}}$")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$f$")
    ax.set_title(f"Fig. 4(g-j) style: N = {N}")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig
