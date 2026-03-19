"""
Entanglement section: Fig. 6(a-d).
CACHED PAPER RESULT: n_links_plots.py and e_eff_plots.py use hardcoded precomputed data
from qv_vs_t_cmpl_A/B.py (which runs Entanglement.py + thresholding).
We use the same hardcoded values for reproducibility.
"""
import numpy as np
import matplotlib.pyplot as plt

# CACHED: From n_links_plots.py and e_eff_plots.py — 4 (t_meas, t_cmpl) combinations
N = [2, 4, 8, 16]

# Dataset key: (t_meas_s, t_cmpl_s) for lookup
_DATASETS = {
    (2e-4, 1e-6): {
        "a_num": np.array([10403, 9379, 8207, 5998]),
        "b_num": np.array([14641, 0, 0, 0]),
        "a_e_avg": np.array([0.1862942603171783, 0.19019864458852706, 0.22976180602164323, 0.2963147865540428]),
        "b_e_avg": np.array([0.2916677341713093, 0.9359266, 1, 1]),
    },
    (2e-4, 1e-5): {
        "a_num": np.array([10231, 9221, 8029, 5857]),
        "b_num": np.array([14641, 0, 0, 0]),
        "a_e_avg": np.array([0.18986770883242557, 0.19106734039838802, 0.23047830018185766, 0.29912289777987533]),
        "b_e_avg": np.array([0.2970211091994934, 0.937134, 1, 1]),
    },
    (8e-4, 1e-6): {
        "a_num": np.array([0, 573, 930, 441]),
        "b_num": np.array([0, 0, 0, 0]),
        "a_e_avg": np.array([1.0, 0.3930422557816903, 0.35708101009412, 0.3936174722932882]),
        "b_e_avg": np.array([0.9997, 0.9999945, 1, 1]),
    },
    (8e-4, 1e-5): {
        "a_num": np.array([0, 552, 926, 447]),
        "b_num": np.array([0, 0, 0, 0]),
        "a_e_avg": np.array([1.0, 0.3937116264805626, 0.356472854011253, 0.39186873473476886]),
        "b_e_avg": np.array([0.9997, 0.999989, 1, 1]),
    },
}

# Paper default (t_meas=8e-4, t_cmpl=1e-5)
a_num = _DATASETS[(8e-4, 1e-5)]["a_num"]
b_num = _DATASETS[(8e-4, 1e-5)]["b_num"]
a_e_avg = _DATASETS[(8e-4, 1e-5)]["a_e_avg"]
b_e_avg = _DATASETS[(8e-4, 1e-5)]["b_e_avg"]


def _get_dataset(t_meas: float, t_cmpl: float) -> dict:
    """Return cached dataset for given (t_meas, t_cmpl). Snaps to nearest cached combo."""
    t_meas_opt = 2e-4 if t_meas < 5e-4 else 8e-4
    t_cmpl_opt = 1e-6 if t_cmpl < 7.5e-6 else 1e-5
    return _DATASETS[(t_meas_opt, t_cmpl_opt)]


def fig6_n_links() -> plt.Figure:
    """
    Fig. 6(a-b): n_links vs N for sequences A and B.
    CACHED PAPER RESULT — from n_links_plots.py.
    """
    colors = plt.cm.plasma(np.linspace(0, 0.7, 2))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N, a_num, "o-", color=colors[0], label=r"$\mathcal{A}$", linewidth=1.8)
    ax.plot(N, b_num, "o--", color=colors[0], label=r"$\mathcal{B}$", linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$n_{\mathrm{links}}$")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=14)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def fig6_epsilon_eff() -> plt.Figure:
    """
    Fig. 6(c-d): ε_eff vs N for sequences A and B.
    CACHED PAPER RESULT — from e_eff_plots.py.
    """
    return fig6_epsilon_eff_interactive(8e-4, 1e-5)


def fig6_n_links_interactive(t_meas: float = 8e-4, t_cmpl: float = 1e-5) -> plt.Figure:
    """
    Fig. 6(a-b): n_links vs N — interactive with t_meas, t_cmpl.
    """
    d = _get_dataset(t_meas, t_cmpl)
    colors = plt.cm.plasma(np.linspace(0, 0.7, 2))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N, d["a_num"], "o-", color=colors[0], label=r"$\mathcal{A}$", linewidth=1.8)
    ax.plot(N, d["b_num"], "o--", color=colors[0], label=r"$\mathcal{B}$", linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$n_{\mathrm{links}}$")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=14)
    ax.set_title(f"t_meas={t_meas:.0e} s, t_cmpl={t_cmpl:.0e} s")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def fig6_epsilon_eff_interactive(t_meas: float = 8e-4, t_cmpl: float = 1e-5) -> plt.Figure:
    """
    Fig. 6(c-d): ε_eff vs N — interactive with t_meas, t_cmpl.
    """
    d = _get_dataset(t_meas, t_cmpl)
    e_th = 0.5
    colors = plt.cm.plasma(np.linspace(0, 0.7, 2))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N, d["a_e_avg"], "o-", color=colors[0], label=r"$\mathcal{A}$", linewidth=1.8)
    ax.plot(N, d["b_e_avg"], "o--", color=colors[0], label=r"$\mathcal{B}$", linewidth=1.8)
    ax.axhspan(e_th, 1.0, facecolor="none", hatch="//", edgecolor=colors[1], alpha=0.6)
    ax.axhline(y=e_th, color=colors[1], linestyle="--", linewidth=1.8, label=r"$\varepsilon_{\mathrm{th}}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{eff}}$")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=14)
    ax.set_title(f"t_meas={t_meas:.0e} s, t_cmpl={t_cmpl:.0e} s")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig
