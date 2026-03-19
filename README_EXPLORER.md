# Programmable Quantum Matter — Interactive Paper Explorer

An interactive Marimo app companion to the paper:

**"Programmable Quantum Matter: Heralding Large Cluster States in Driven Inhomogeneous Spin Ensembles"**  
*Physics Review X Quantum (accepted)*

## Quick start

```bash
# Install dependencies
pip install marimo numpy matplotlib pandas scipy

# Run in edit mode (development)
marimo edit paper_explorer.py

# Run as app (code hidden, polished view)
marimo run paper_explorer.py
```

Then open the URL shown in the terminal (typically http://127.0.0.1:2718).

## What this app does

- **Section 0 — Landing:** Title, plain-language and technical summaries, "Why this matters," main contributions
- **Section 1 — Architecture:** Big picture, two ensembles, global drive, compilation pipeline
- **Section 2 — Physical system:** SiV, strain driving, amplitude/detuning errors, heterogeneity as resource
- **Section 3 — SAFE-GRAPE:** Composite pulses, rCinBB vs SAFE-GRAPE, robustness over (ε, f)
- **Section 4 — Filter function:** Dynamical decoupling, sequence A vs B, noise filtering
- **Section 5 — Thermal/T₂:** Heating vs coherence tradeoff, optimal N, interactive explorer
- **Section 6 — Entanglement:** Link statistics, n_links, ε_eff
- **Section 7 — Compilation:** Link statistics h_j, strain window, Algorithm II
- **Section 8 — Figure explorer:** Provenance, source files, reproducibility
- **Section 9 — Reproducibility:** What can be reproduced and how
- **Section 10 — Limitations:** Assumptions, simulated vs experimental, future directions

## Figure reproducibility

| Figure | Source | How to reproduce |
|-------|--------|-------------------|
| Fig. 2 | SAFE_GRAPE.ipynb | Run notebook (PyTorch) |
| Fig. 3 | Filter_Function_I.ipynb | Run notebook |
| Fig. 4 | Decoherence_sim.py, T2_heatmaps.py | Requires CSV from Filter_Function_II.py |
| Fig. 6 | n_links_plots.py, e_eff_plots.py | Cached in app |
| Fig. 8 | Alg_II_heatmap.py, Alg_II_linkstat.py | Live in app |
| Fig. 9 | strain_window_alg1.py | Live (needs qutip) or schematic |
| Fig. 11 | dil_fridge.py | Live in app |

## CSV files for Fig. 4

Fig. 4 (T₂ vs N, T₂ heatmaps) requires precomputed CSV files from `Filter_Function_II.py`:

1. `cd "Dynamical Decoupling - Filter Function"`
2. Run `Filter_Function_II.py` (or the parallel version for N=2,4,8,16)
3. This produces `N=2_results_parallel.csv`, etc. in that folder
4. Copy or symlink these into `Temperature and T2 simulations` (they may already exist)

The app will use them automatically when available.

## Project structure

```
Programmable_Quantum_Matter/
├── paper_explorer.py      # Main Marimo app
├── app/
│   ├── paths.py           # Repo path resolution
│   ├── data_loader.py     # CSV loading
│   ├── heat_capacity.py   # Diamond heat capacity (from paper)
│   ├── config.py          # Fast/full mode
│   └── figures/           # Figure generation
│       ├── compilation.py  # Fig. 8, 9
│       ├── decoherence.py  # Fig. 4
│       ├── entanglement.py # Fig. 6
│       └── heat_modeling.py # Fig. 11
├── requirements.txt
└── README_EXPLORER.md
```

## License

Same as the parent repository (MIT).
