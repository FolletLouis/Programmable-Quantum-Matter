"""
Strain window analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import norm
from qutip import Qobj


# Physical constants
DELTA_ZPL_0 = 406.8e12      # Hz, zero-phonon line frequency
LAMBDA_SO_GS = 46e9         # Hz, ground state spin-orbit splitting  
LAMBDA_SO_ES = 255e9        # Hz, excited state spin-orbit splitting
GAMMA_L = 1.4e9             # Hz/T, orbital g-factor
GAMMA_S = 14e9              # Hz/T, spin g-factor
B_FIELD = 0.17              # T, magnetic field magnitude
B_Z = np.sqrt(0.5) * B_FIELD
B_X = np.sqrt(0.5) * B_FIELD

# Strain coupling parameters (Hz/strain)
D_GS = 1.3e15
F_GS = -1.7e15
D_ES = 1.8e15 
F_ES = -3.4e15
T_PARALLEL = -1.7e15
T_ORTHOGONAL = 0.078e15

# Analysis parameters
LASER_FREQ = 406.711e12     # Hz
STRAIN_RANGE = 9e-5
N_POINTS = 400
N_QUANTILES = 10
SIGMA_WIDTH = 6e-5          # Meesala width


def strain_shift_gs(eps_xx=0, eps_yy=0, eps_yz=0):
    """Ground state strain-induced energy shift."""
    return D_GS * (eps_xx - eps_yy) + F_GS * eps_yz


def strain_shift_es(eps_xx=0, eps_yy=0, eps_yz=0):
    """Excited state strain-induced energy shift.""" 
    return D_ES * (eps_xx - eps_yy) + F_ES * eps_yz


def zpl_frequency(eps_xx=0, eps_yy=0):
    """Zero-phonon line frequency with strain correction."""
    return DELTA_ZPL_0 + T_ORTHOGONAL * (eps_xx + eps_yy)


def compute_transitions(eps_yy):
    """
    Compute optical transition frequencies for given strain.
    
    Parameters
    ----------
    eps_yy : float
        Strain component along y-axis
        
    Returns
    -------
    dict
        Dictionary containing energy levels and transition frequencies
    """
    # Strain-induced shifts
    shift_gs = D_GS * (-eps_yy)
    shift_es = D_ES * (-eps_yy) 
    delta_zpl = zpl_frequency(eps_yy=eps_yy)
    
    # Ground state Hamiltonian
    H_gs = Qobj(np.array([
        [-LAMBDA_SO_GS/2 - GAMMA_L*B_Z - GAMMA_S*B_Z, 0, shift_gs, GAMMA_S*B_X],
        [0, -LAMBDA_SO_GS/2 + GAMMA_L*B_Z + GAMMA_S*B_Z, GAMMA_S*B_X, shift_gs],
        [shift_gs, GAMMA_S*B_X, LAMBDA_SO_GS/2 + GAMMA_L*B_Z - GAMMA_S*B_Z, 0],
        [GAMMA_S*B_X, shift_gs, 0, LAMBDA_SO_GS/2 - GAMMA_L*B_Z + GAMMA_S*B_Z]
    ]))
    
    # Excited state Hamiltonian  
    H_es = Qobj(np.array([
        [-LAMBDA_SO_ES/2 - GAMMA_L*B_Z - GAMMA_S*B_Z, 0, shift_es, GAMMA_S*B_X],
        [0, -LAMBDA_SO_ES/2 + GAMMA_L*B_Z + GAMMA_S*B_Z, GAMMA_S*B_X, shift_es],
        [shift_es, GAMMA_S*B_X, LAMBDA_SO_ES/2 + GAMMA_L*B_Z - GAMMA_S*B_Z, 0],
        [GAMMA_S*B_X, shift_es, 0, LAMBDA_SO_ES/2 - GAMMA_L*B_Z + GAMMA_S*B_Z]
    ])) + Qobj(delta_zpl * np.eye(4))
    
    # Compute eigenvalues
    evals_gs, _ = H_gs.eigenstates()
    evals_es, _ = H_es.eigenstates()
    
    # Optical transitions
    c2_transition = evals_es[0] - evals_gs[0]  
    c3_transition = evals_es[1] - evals_gs[1]
    
    return {
        'E_g_minus_down': evals_gs[0],
        'E_g_plus_up': evals_gs[1], 
        'E_u_minus_down': evals_es[0],
        'E_u_plus_up': evals_es[1],
        'C2_transition': c2_transition,
        'C3_transition': c3_transition
    }


def find_zero_crossings(y, x, threshold=0.2):
    """
    Find zero crossings of detuning curves.
    
    Parameters
    ----------
    y : array_like
        Detuning values
    x : array_like  
        Strain values
    threshold : float
        Threshold for considering valid crossings (GHz)
        
    Returns
    -------
    list
        List of (crossing_position, slope_sign) tuples
    """
    sign_changes = np.where(np.diff(np.signbit(y)))[0]
    crossings = []
    
    for i in sign_changes:
        if abs(y[i]) > threshold and abs(y[i+1]) > threshold:
            continue
            
        # Linear interpolation for crossing position
        frac = -y[i] / (y[i+1] - y[i])
        x_cross = x[i] + frac * (x[i+1] - x[i])
        
        # Calculate slope
        if 0 < i < len(y) - 1:
            slope = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
        else:
            slope = y[i+1] - y[i]
            
        crossings.append((x_cross, np.sign(slope)))
        
    return crossings


def main():
    """Generate strain window analysis plot."""
    
    # Set up strain grid
    eps_dc = np.linspace(-STRAIN_RANGE, STRAIN_RANGE, N_POINTS)
    
    # Generate bias strain values using quantile sampling
    quantiles = (np.arange(1, N_QUANTILES + 1) - 0.5) / N_QUANTILES
    bias_strains = SIGMA_WIDTH * norm.ppf(quantiles)
    bias_strains = np.append(bias_strains, 8.3e-5)
    
    # Compute detuning matrix
    detuning_matrix = []
    for bias in bias_strains:
        transitions = [compute_transitions(eps + bias)['C2_transition'] 
                      for eps in eps_dc]
        detuning = (np.array(transitions) - LASER_FREQ) * 1e-9  # Convert to GHz
        detuning_matrix.append(detuning)
    detuning_matrix = np.array(detuning_matrix)
    
    # Set up plotting
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"], 
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
    })
    
    # Color mapping
    color_norm = Normalize(vmin=bias_strains.min() * 1e5, 
                          vmax=bias_strains.max() * 1e5)
    colormap = mpl.cm.plasma
    scalar_map = ScalarMappable(norm=color_norm, cmap=colormap)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 2.6))
    
    # Find valid crossing points
    valid_crossings = []
    crossing_colors = []
    
    for bias, detuning in zip(bias_strains, detuning_matrix):
        color = colormap(color_norm(bias * 1e5))
        ax.plot(eps_dc * 1e5, detuning, linewidth=0.9, color=color)
        
        for x_cross, slope_sign in find_zero_crossings(detuning, eps_dc):
            if slope_sign > 0:  # Only positive slope crossings
                valid_crossings.append(x_cross)
                crossing_colors.append(color)
    
    # Remove spurious points and define strain window
    valid_crossings = np.array(valid_crossings[2:])
    crossing_colors = crossing_colors[2:]
    
    strain_window_left = valid_crossings.min()
    strain_window_right = valid_crossings.max()
    
    # Plot strain window and crossing points
    ax.axvspan(strain_window_left * 1e5, strain_window_right * 1e5, 
               color='grey', alpha=0.15, zorder=0)
    
    for x_cross, color in zip(valid_crossings, crossing_colors):
        ax.plot(x_cross * 1e5, 0, marker='o', markersize=3.5,
                color=color, markeredgecolor='black', zorder=3)
    
    # Formatting
    ax.axhline(0, linestyle='--', linewidth=0.8, color='k')
    ax.text(0.02, 0.62, r'$f_L$', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9)
    
    ax.set_xlabel(r'Global strain $\epsilon^{\mathrm{dc}}_{E_{gx}}$ ($\times 10^{-5}$)')
    ax.set_ylabel(r'$\Delta f = f_{\mathrm{opt}} - f_L$ (GHz)')
    ax.set_xlim(eps_dc[0] * 1e5, eps_dc[-1] * 1e5)
    ax.set_ylim(-20, 13)
    
    # Colorbar
    cbar = plt.colorbar(scalar_map, ax=ax, pad=0.02)
    cbar.set_label(r'$\epsilon^{\mathrm{bias}}_{E_{gx}}$ ($\times 10^{-5}$)')
    
    plt.tight_layout()
    fig.savefig("strain_window.pdf", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
