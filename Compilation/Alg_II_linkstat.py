import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # optional, if you want an integer locator
import matplotlib.cm as cm
import matplotlib as mpl

# Ensure font is Arial and text is not converted to paths
mpl.rcParams.update({
    'font.size': 21,
    'font.family': 'Arial',
    'pdf.fonttype': 42  # Ensures fonts are stored as text, not outlines
})


c = plt.cm.plasma(np.linspace(0, 1, 5))

def system_A(n1, N):
    s = []
    for n in n1:
        mod_val = n % (2 * N)
        if mod_val == 0:
            s.append(1)
        elif mod_val <= N:
            s.append(mod_val)
        else:  # mod_val > N
            s.append(2 * N + 1 - mod_val)
    return s

# Fixed parameters for the two systems (the "pattern" parameters)

#Convention: Na > Nb
Na = 5 # number of distinct labels for system A (assume labels 1,...,Na)
Nb = 4 # for system B

# Define a range of p_max to sweep over.
# For example, let p_max range from 10 to 50 in steps of 5.
p_max_values = np.arange(1, 50, 1) # 10,15,...,50


# Prepare lists to store the fractions for each category for each p_max.
hit_zero_list = []
hit_one_list = []
hit_two_list = []
hit_above_list = []

scaling = int(1)

# Total number of (i,j) pairs
total_pairs = Na * Nb

for p_max in p_max_values:
    # Create p from 1 to p_max
    p_a = np.arange(1, p_max+1)
    p_b = np.arange(1, p_max*scaling+1)
    # Compute sequence outputs for system A and system B for given N's.
    y_a = system_A(p_a, Na)
    y_b = system_A(p_b, Nb)
    
    # Build "links" list: pair [y_a[i], y_b[i]] for i in index range.
    links = []
    for i in range(len(p_a)):
        links.append([y_a[i], y_b[i*scaling]])
    
    # Count occurrences for each pair (i,j) with i=1,...,Na and j=1,...,Nb.
    occ = []
    # Loop over all possible label combinations:
    for i in range(Na):       # i = 0,...,Na-1 corresponds to labels 1,...,Na
        for j in range(Nb):   # j = 0,...,Nb-1 corresponds to labels 1,...,Nb
            # Count how many times the pair [i+1, j+1] occurs in links.
            count = sum(1 for link in links if link == [i+1, j+1])
            occ.append(count)
    
    # Initialize hit counters:
    hit_zero = 0
    hit_one = 0
    hit_two = 0
    hit_above = 0

    
    # Use proper if-elif chain so that each occ value falls into one category.
    for o in occ:
        if o == 0:
            hit_zero += 1
        elif o == 1:
            hit_one += 1
        elif o == 2:
            hit_two += 1
        elif o > 2:
            hit_above += 1
    
    # Compute relative frequencies (divide by total pairs)
    hit_zero_list.append(hit_zero / total_pairs)
    hit_one_list.append(hit_one / total_pairs)
    hit_two_list.append(hit_two / total_pairs)
    hit_above_list.append(hit_above / total_pairs)


# 1) square figure
fig, ax = plt.subplots(figsize=(9, 8))

# 2) your four curves
ax.plot(p_max_values, hit_zero_list, '-o', color = c[0], label=r'$h_{0}$')
ax.plot(p_max_values, hit_one_list, '-o', color = c[1], label=r'$h_{1}$')
ax.plot(p_max_values, hit_two_list, '-o', color = c[2], label=r'$h_{2}$')
ax.plot(p_max_values, hit_above_list, '-o', color = c[3], label=r'$h_{>2}$')

# labels, legend, grid
ax.set_xlabel(r'$t$ (in units of $T_{A}$)')
ax.set_ylabel('Link Statistics')
ax.legend(
    fontsize=16,         # 🔽 Smaller font size for legend
    framealpha=0.,      # Optional: semi-transparent background
    facecolor='white'    # Optional: legend box color
)
ax.grid(True, which='both', ls='--', lw=0.5)

# 3) equal data‐unit aspect
#ax.set_aspect('equal', adjustable='box')

# save & show
fig.tight_layout()
fig.savefig('Fig8_d.pdf')
plt.show()
