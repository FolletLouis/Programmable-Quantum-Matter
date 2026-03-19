"""
Compilation section figures: Fig. 8 (a-c, d) and Fig. 9.
EXACT-PAPER REPRODUCTION: Logic from Alg_II_heatmap.py, Alg_II_linkstat.py, strain_window_alg1.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Tuple

# --- Fig 8 heatmap logic (from Alg_II_heatmap.py) ---
def system_A(n1: np.ndarray, N: int) -> list:
    """Sawtooth pattern for system A."""
    s = []
    for n in n1:
        mod_val = n % (2 * N)
        if mod_val == 0:
            s.append(1)
        elif mod_val <= N:
            s.append(mod_val)
        else:
            s.append(2 * N + 1 - mod_val)
    return s


def evaluate_occurrences(N: int, M: int, p_max: int, scaling: int) -> np.ndarray:
    """Compute occurrence matrix for link statistics."""
    p_a = np.arange(1, p_max + 1)
    p_b = np.arange(1, p_max * scaling + 1)
    y_a = system_A(p_a, N)
    y_b = system_A(p_b, M)
    links = []
    for i in range(len(p_a)):
        index_b = i * scaling
        if index_b < len(y_b):
            links.append([y_a[i], y_b[index_b]])
    occ = np.zeros((N, M), dtype=int)
    for link in links:
        i_label, j_label = link[0] - 1, link[1] - 1
        if i_label < N and j_label < M:
            occ[i_label, j_label] += 1
    return occ


def find_min_solution(N: int, M: int, pmax_min: int, pmax_max: int,
                      scaling_min: int, scaling_max: int) -> Tuple:
    """Find best (S, p_max) for minimal hit-zero fraction."""
    best_hit_zero, best_S, best_pmax, best_occ = 1.0, None, None, None
    for S in range(scaling_min, scaling_max + 1):
        for p_max in range(pmax_min, pmax_max + 1):
            occ = evaluate_occurrences(N, M, p_max, S)
            frac = np.sum(occ == 0) / (N * M)
            if frac == 0:
                return S, p_max, frac, occ
            if frac < best_hit_zero:
                best_hit_zero, best_S, best_pmax, best_occ = frac, S, p_max, occ
    return best_S, best_pmax, best_hit_zero, best_occ


def fig8_heatmaps(Na_max: int = 10, Nb_max: int = 10) -> plt.Figure:
    """
    Fig. 8(a-c): Heatmaps of min{h0}, argmin{h0}, J.
    EXACT-PAPER REPRODUCTION from Alg_II_heatmap.py.
    Mirror logic requires Nb in Na_values and Na in Nb_values when Na < Nb.
    Use symmetric grid so mirror lookup always works.
    """
    n_max = max(Na_max, Nb_max)
    Na_values = np.arange(1, n_max + 1)
    Nb_values = np.arange(1, n_max + 1)
    scaling_min, scaling_max = 1, 10
    scaling_mat = np.full((len(Na_values), len(Nb_values)), np.nan)
    pmax_mat = np.full((len(Na_values), len(Nb_values)), np.nan)
    hit_zero_mat = np.full((len(Na_values), len(Nb_values)), np.nan)

    for i, Na in enumerate(Na_values):
        for j, Nb in enumerate(Nb_values):
            if Na >= Nb:
                S, p_max, frac, _ = find_min_solution(Na, Nb, 1, Na * Nb, scaling_min, Nb)
                scaling_mat[i, j] = S if S is not None else np.nan
                pmax_mat[i, j] = p_max if p_max is not None else np.nan
                hit_zero_mat[i, j] = frac
            else:
                idx_i = np.where(Na_values == Nb)[0][0]
                idx_j = np.where(Nb_values == Na)[0][0]
                scaling_mat[i, j] = scaling_mat[idx_i, idx_j]
                pmax_mat[i, j] = pmax_mat[idx_i, idx_j]
                hit_zero_mat[i, j] = hit_zero_mat[idx_i, idx_j]

    scaling_mat = scaling_mat[:Na_max, :Nb_max]
    pmax_mat = pmax_mat[:Na_max, :Nb_max]
    hit_zero_mat = hit_zero_mat[:Na_max, :Nb_max]
    Na_values = np.arange(1, Na_max + 1)
    Nb_values = np.arange(1, Nb_max + 1)

    extent = [Nb_values[0] - 0.5, Nb_values[-1] + 0.5, Na_values[0] - 0.5, Na_values[-1] + 0.5]
    plots = [
        (hit_zero_mat, r"min$_m\{h_0\}$"),
        (scaling_mat, r"argmin$_m\{h_0\}$"),
        (pmax_mat, r"$\mathcal{J}$ (in units of $T_{\mathcal{A}}$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (mat, label) in zip(axes, plots):
        im = ax.imshow(mat, origin="lower", extent=extent, aspect="equal", cmap="plasma")
        ax.set_xlabel(r"$N_b$")
        ax.set_ylabel(r"$N_a$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def fig8_linkstat(Na: int = 5, Nb: int = 4) -> plt.Figure:
    """
    Fig. 8d: Link statistics h_j vs time window.
    EXACT-PAPER REPRODUCTION.
    """
    def system_A_local(n1, N):
        s = []
        for n in n1:
            mod_val = n % (2 * N)
            s.append(1 if mod_val == 0 else (mod_val if mod_val <= N else 2 * N + 1 - mod_val))
        return s

    p_max_values = np.arange(1, 50, 1)
    scaling = 1
    total_pairs = Na * Nb
    hit_zero_list, hit_one_list, hit_two_list, hit_above_list = [], [], [], []

    for p_max in p_max_values:
        p_a = np.arange(1, p_max + 1)
        p_b = np.arange(1, p_max * scaling + 1)
        y_a = system_A_local(p_a, Na)
        y_b = system_A_local(p_b, Nb)
        links = [[y_a[i], y_b[i * scaling]] for i in range(len(p_a)) if i * scaling < len(y_b)]
        occ = [sum(1 for link in links if link == [i + 1, j + 1])
               for i in range(Na) for j in range(Nb)]
        hit_zero = sum(1 for o in occ if o == 0) / total_pairs
        hit_one = sum(1 for o in occ if o == 1) / total_pairs
        hit_two = sum(1 for o in occ if o == 2) / total_pairs
        hit_above = sum(1 for o in occ if o > 2) / total_pairs
        hit_zero_list.append(hit_zero)
        hit_one_list.append(hit_one)
        hit_two_list.append(hit_two)
        hit_above_list.append(hit_above)

    fig, ax = plt.subplots(figsize=(8, 6))
    c = plt.cm.plasma(np.linspace(0, 1, 5))
    ax.plot(p_max_values, hit_zero_list, "-o", color=c[0], label=r"$h_0$")
    ax.plot(p_max_values, hit_one_list, "-o", color=c[1], label=r"$h_1$")
    ax.plot(p_max_values, hit_two_list, "-o", color=c[2], label=r"$h_2$")
    ax.plot(p_max_values, hit_above_list, "-o", color=c[3], label=r"$h_{>2}$")
    ax.set_xlabel(r"$t$ (in units of $T_{\mathcal{A}}$)")
    ax.set_ylabel("Link Statistics")
    ax.legend(fontsize=12)
    ax.grid(True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def fig9_strain_window(
    strain_range: float = 9e-5,
    sigma_width: float = 6e-5,
    n_quantiles: int = 10,
    n_points: int = 150,
) -> plt.Figure:
    """
    Fig. 9: Strain window for SiV. Source: strain_window_alg1.py.
    FULL REPRODUCTION requires qutip; if unavailable, returns schematic.
    """
    try:
        from scipy.stats import norm
        from qutip import Qobj
    except ImportError:
        # SIMPLIFIED: schematic when qutip not available
        fig, ax = plt.subplots(figsize=(6, 4))
        eps = np.linspace(-strain_range, strain_range, 100) * 1e5
        ax.plot(eps, 5 * np.sin(eps * 0.5) - 2, "b-", label="Detuning (schematic)")
        ax.axhline(0, linestyle="--", color="k")
        ax.axvspan(-3, 3, color="grey", alpha=0.15)
        ax.set_xlabel(r"Global strain $\epsilon^{\mathrm{dc}}$ ($\times 10^{-5}$)")
        ax.set_ylabel(r"$\Delta f$ (GHz)")
        ax.set_title("Fig. 9 (simplified): Strain window — requires qutip for full reproduction")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return fig

    # Full reproduction from strain_window_alg1.py
    DELTA_ZPL_0 = 406.8e12
    LAMBDA_SO_GS, LAMBDA_SO_ES = 46e9, 255e9
    GAMMA_L, GAMMA_S = 1.4e9, 14e9
    B_FIELD = 0.17
    B_Z = B_X = np.sqrt(0.5) * B_FIELD
    D_GS, F_GS = 1.3e15, -1.7e15
    D_ES, F_ES = 1.8e15, -3.4e15
    T_ORTHOGONAL = 0.078e15
    LASER_FREQ = 406.711e12

    def zpl_frequency(eps_yy):
        return DELTA_ZPL_0 + T_ORTHOGONAL * (2 * eps_yy)

    def compute_transitions(eps_yy):
        shift_gs = D_GS * (-eps_yy)
        shift_es = D_ES * (-eps_yy)
        delta_zpl = zpl_frequency(eps_yy)
        H_gs = Qobj(np.array([
            [-LAMBDA_SO_GS/2 - GAMMA_L*B_Z - GAMMA_S*B_Z, 0, shift_gs, GAMMA_S*B_X],
            [0, -LAMBDA_SO_GS/2 + GAMMA_L*B_Z + GAMMA_S*B_Z, GAMMA_S*B_X, shift_gs],
            [shift_gs, GAMMA_S*B_X, LAMBDA_SO_GS/2 + GAMMA_L*B_Z - GAMMA_S*B_Z, 0],
            [GAMMA_S*B_X, shift_gs, 0, LAMBDA_SO_GS/2 - GAMMA_L*B_Z + GAMMA_S*B_Z]
        ]))
        H_es = Qobj(np.array([
            [-LAMBDA_SO_ES/2 - GAMMA_L*B_Z - GAMMA_S*B_Z, 0, shift_es, GAMMA_S*B_X],
            [0, -LAMBDA_SO_ES/2 + GAMMA_L*B_Z + GAMMA_S*B_Z, GAMMA_S*B_X, shift_es],
            [shift_es, GAMMA_S*B_X, LAMBDA_SO_ES/2 + GAMMA_L*B_Z - GAMMA_S*B_Z, 0],
            [GAMMA_S*B_X, shift_es, 0, LAMBDA_SO_ES/2 - GAMMA_L*B_Z + GAMMA_S*B_Z]
        ])) + Qobj(delta_zpl * np.eye(4))
        evals_gs, _ = H_gs.eigenstates()
        evals_es, _ = H_es.eigenstates()
        return evals_es[0] - evals_gs[0]

    eps_dc = np.linspace(-strain_range, strain_range, n_points)
    quantiles = (np.arange(1, n_quantiles + 1) - 0.5) / n_quantiles
    bias_strains = np.append(sigma_width * norm.ppf(quantiles), 8.3e-5)
    detuning_matrix = np.array([
        [(compute_transitions(eps + bias) - LASER_FREQ) * 1e-9 for eps in eps_dc]
        for bias in bias_strains
    ])
    color_norm = plt.Normalize(vmin=bias_strains.min() * 1e5, vmax=bias_strains.max() * 1e5)
    colormap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for bias, detuning in zip(bias_strains, detuning_matrix):
        ax.plot(eps_dc * 1e5, detuning, linewidth=0.9, color=colormap(color_norm(bias * 1e5)))
    ax.axhline(0, linestyle="--", color="k")
    ax.set_xlabel(r"Global strain $\epsilon^{\mathrm{dc}}_{E_{gx}}$ ($\times 10^{-5}$)")
    ax.set_ylabel(r"$\Delta f = f_{\mathrm{opt}} - f_L$ (GHz)")
    ax.set_xlim(eps_dc[0] * 1e5, eps_dc[-1] * 1e5)
    ax.set_ylim(-20, 13)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=colormap), ax=ax)
    cbar.set_label(r"$\epsilon^{\mathrm{bias}}_{E_{gx}}$ ($\times 10^{-5}$)")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig
