import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib as mpl

# Ensure font is Arial and text is not converted to paths
mpl.rcParams.update({
    'font.size': 21,
    'font.family': 'Arial',
    'pdf.fonttype': 42  # Ensures fonts are stored as text, not outlines
})

def system_A(n1, N):
    """
    Return a sequence of labels for each element in n1 following a 
    sawtooth (up-then-down) pattern for a given N.
    """
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

def evaluate_occurrences(N, M, p_max, scaling):
    """
    For system A (with N labels) and system B (with M labels), 
    generate:
      p_a = [1,..., p_max] and p_b = [1,..., p_max*scaling].
    Compute y_a = system_A(p_a, N) and y_b = system_A(p_b, M).
    Then, for each index i from 0 to p_max-1, pair y_a[i] with y_b[i*scaling]
    (provided that index is valid).
    Returns the occurrence matrix occ of shape (N, M), where occ[i,j] counts
    how many times the pair (i+1,j+1) appears.
    """
    p_a = np.arange(1, p_max+1)
    p_b = np.arange(1, p_max*scaling+1)
    y_a = system_A(p_a, N)
    y_b = system_A(p_b, M)
    
    links = []
    for i in range(len(p_a)):
        index_b = i * scaling
        if index_b < len(y_b):
            links.append([y_a[i], y_b[index_b]])
    
    occ = np.zeros((N, M), dtype=int)
    for link in links:
        i_label = link[0] - 1  # convert to 0-index
        j_label = link[1] - 1
        if i_label < N and j_label < M:
            occ[i_label, j_label] += 1
    return occ

def find_min_solution(N, M, pmax_min, pmax_max, scaling_min, scaling_max):
    """
    For a given (N, M), search for an integer scaling (S) between scaling_min and scaling_max
    and p_max between pmax_min and pmax_max such that every label pair occurs at least once.
    If a solution is found with hit-zero fraction = 0, return it immediately.
    Otherwise, return the (S, p_max) combination which minimizes the hit-zero fraction.
    
    Returns: (best_S, best_pmax, best_hit_zero, occ)
    where best_hit_zero is the fraction of label pairs (i,j) with 0 occurrences.
    """
    best_hit_zero = 1.0
    best_S = None
    best_pmax = None
    best_occ = None
    for S in range(scaling_min, scaling_max + 1):
        for p_max in range(pmax_min, pmax_max + 1):
            occ = evaluate_occurrences(N, M, p_max, S)
            hit_zero = np.sum(occ == 0)
            frac = hit_zero / (N * M)
            # If the condition is perfectly met, return immediately.
            if frac == 0:
                return S, p_max, frac, occ
            if frac < best_hit_zero:
                best_hit_zero = frac
                best_S = S
                best_pmax = p_max
                best_occ = occ
    return best_S, best_pmax, best_hit_zero, best_occ

# --- Set up grids for Na and Nb.
# We assume Na and Nb are the number of distinct labels for system A and B.
# Convention: we will simulate for Na >= Nb and mirror for Na < Nb.
Na_values = np.arange(1, 11)  # e.g., 3 to 10
Nb_values = np.arange(1, 11)  # e.g., 2 to 10

# Define search bounds:
pmax_lower = lambda Na, Nb: 2 * Na * Nb  # minimal p_max must be at least 2*Na*Nb.
pmax_max = 200  # maximum p_max in our search.
scaling_min = 1
scaling_max = 10

# Prepare result matrices.
scaling_mat = np.full((len(Na_values), len(Nb_values)), np.nan)
pmax_mat = np.full((len(Na_values), len(Nb_values)), np.nan)
hit_zero_mat = np.full((len(Na_values), len(Nb_values)), np.nan)

# Loop over combinations.
for i, Na in enumerate(Na_values):
    for j, Nb in enumerate(Nb_values):
        if Na >= Nb:
            pmax_min_val = pmax_lower(Na, Nb)
            S, p_max, frac, occ = find_min_solution(Na, Nb, 1, Na*Nb, scaling_min, Nb)
            scaling_mat[i, j] = S if S is not None else np.nan
            pmax_mat[i, j] = p_max if p_max is not None else np.nan
            hit_zero_mat[i, j] = frac
        else:
            # Mirror the result for Na < Nb by swapping indices:
            # (Na, Nb) result = result (Nb, Na)
            idx_i = np.where(Na_values == Nb)[0][0]
            idx_j = np.where(Nb_values == Na)[0][0]
            scaling_mat[i, j] = scaling_mat[idx_i, idx_j]
            pmax_mat[i, j] = pmax_mat[idx_i, idx_j]
            hit_zero_mat[i, j] = hit_zero_mat[idx_i, idx_j]

# --- Common settings
extent = [Nb_values[0]-0.5, Nb_values[-1]+0.5,
          Na_values[0]-0.5, Na_values[-1]+0.5]

plots = [
    ("Fig8_a.pdf", hit_zero_mat,    "plasma", r"min$_{m}\{h₀\}$"),
    ("Fig8_b.pdf",   scaling_mat,    "plasma",  r"argmin$_m\{h₀\}$"),
    ("Fig8_c.pdf",      pmax_mat,       "plasma",   r"$\mathcal{J}$ (in units of $T_{A}$)"),
]

for fname, mat, cmap, cbar_label in plots:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mat,
                   origin='lower',
                   extent=extent,
                   aspect='equal',
                   cmap=cmap)
    ax.set_xlabel(r"$N_{b}$")
    ax.set_ylabel(r"$N_{a}$")
    cbar = fig.colorbar(im, ax=ax,
                        fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

print("Saved:", [p[0] for p in plots])

