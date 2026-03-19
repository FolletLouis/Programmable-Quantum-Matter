# Marimo App Architecture Plan

## Paper: "Programmable Quantum Matter: Heralding Large Cluster States in Driven Inhomogeneous Spin Ensembles"

**PRX Quantum (accepted)**

---

## 1. Proposed File Structure

```
Programmable_Quantum_Matter/
├── paper_explorer.py              # Main marimo app (Sections 0–10)
├── APP_ARCHITECTURE.md            # This document
├── requirements.txt
├── README_EXPLORER.md             # How to run the app
├── PQM_main_text.pdf              # Paper (existing)
│
├── app/
│   ├── __init__.py
│   ├── paths.py                   # Repo path resolution
│   ├── data_loader.py             # CSV loading, cached data
│   ├── heat_capacity.py           # Diamond heat capacity (from repo)
│   ├── config.py                  # Fast vs full mode
│   │
│   ├── figures/                   # Figure generation wrappers
│   │   ├── __init__.py
│   │   ├── safe_grape.py          # Fig. 2, S4 (cached/schematic)
│   │   ├── filter_function.py     # Fig. 3 (cached/schematic)
│   │   ├── decoherence.py         # Fig. 4 (from CSV)
│   │   ├── entanglement.py       # Fig. 6 (cached)
│   │   ├── compilation.py         # Fig. 8, 9
│   │   ├── heat_modeling.py      # Fig. 11, 12
│   │   └── strain_driving.py     # Fig. S2 (schematic)
│   │
│   └── content/                   # Section content generators
│       ├── __init__.py
│       ├── landing.py
│       ├── architecture.py
│       ├── physical_system.py
│       ├── safe_grape.py
│       ├── filter_function.py
│       ├── thermal.py
│       ├── entanglement.py
│       ├── compilation.py
│       └── limitations.py
│
└── [existing repo folders]
    ├── Error-Correcting Pulses_/
    ├── Dynamical Decoupling - Filter Function/
    ├── Temperature and T2 simulations/
    ├── Entanglement simulations/
    ├── Compilation/
    ├── Heat_Modeling/
    └── SiV Strain Driving/
```

---

## 2. Figure/Code Provenance Mapping (from README)

| Figure | Section | Repo File(s) | Type | Notes |
|--------|---------|--------------|------|-------|
| **Fig. 2** | Main I | SAFE_GRAPE.ipynb | Exact | PyTorch, ~minutes |
| **Fig. S4** | Supp | BandwidthAware_SAFEGRAPE.ipynb | Exact | Hardware constraints |
| **Fig. 3(b-d)** | Main II | Filter_Function_I.ipynb | Exact | Uses filter_functions |
| **Fig. 4(a-f)** | Main III | Decoherence_sim.py | Cached | Needs CSV from Filter_Function_II |
| **Fig. 4(g-j)** | Main III | T2_heatmaps.py | Cached | Same CSV |
| **Fig. 13** | Supp | qubit_scaling.py | Cached | m-sweep |
| **Fig. 6(a-d)** | Main IV | n_links_plots.py, e_eff_plots.py | Cached | Via qv_vs_t_cmpl_A/B |
| **Fig. 8(a-c)** | Main V | Alg_II_heatmap.py | Exact | Live regeneration |
| **Fig. 8(d)** | Main V | Alg_II_linkstat.py | Exact | Live |
| **Fig. 9** | Main V | strain_window_alg1.py | Exact | Requires qutip |
| **Fig. 11** | Supp | dil_fridge.py | Exact | Live |
| **Fig. 12** | Supp | temp_siv_fast.py | Exact | Live |
| **Fig. S2** | Supp | StrainDriving.ipynb | Schematic | qutip, Bloch sphere |

**CSV dependency chain:** Filter_Function_II.py → N=2,4,8,16_results_parallel.csv, Fid_N=2.csv → Decoherence_sim, T2_heatmaps, Entanglement, qv_vs_t_cmpl_*

---

## 3. Section-by-Section Marimo App Plan

### SECTION 0 — Landing Page
- **Question:** What is this paper about?
- **Content:** Title, authors (from paper), plain-language summary, technical summary, "Why this matters" callout, "Main contributions" callout
- **Contributions (from spec):**
  - Robust global unitary control under simultaneous amplitude and detuning errors
  - Broadband dynamical decoupling using SAFE-GRAPE-based sequences
  - Improved coherence vs bang-bang/interleaved
  - Reduced thermal burden for global control
  - Improved heralded entanglement-link generation
  - Efficient compilation toward bipartite graph cluster states
- **Method:** Static markdown + mo.callout
- **Result:** Polished landing

### SECTION 1 — The Big Picture / Architecture
- **Question:** What is the proposed architecture?
- **Content:** Two spatially separated heterogeneous spin ensembles, single global strain drive per ensemble, optimized DDS, single-photon entanglement, compilation into bipartite cluster states
- **Method:** mo.accordion for layers, optional schematic (matplotlib)
- **Result:** Architecture overview, optimization objective J(θ), requirements R1–R5
- **Label:** Educational reconstruction (schematic)

### SECTION 2 — Physical System and Control Model
- **Question:** What is the physical system and why does heterogeneity matter?
- **Content:** Group-IV color centers / SiV, strain-driven gates, amplitude error ε_i, off-resonance f_i, heterogeneity as challenge and resource
- **Method:** Two layers (popular + technical), optional Bloch-sphere or parameter plot
- **Result:** Clear explanation of control model
- **Label:** Educational reconstruction (if interactive)

### SECTION 3 — Composite Pulses and SAFE-GRAPE
- **Question:** How do optimized control pulses work?
- **Content:** Composite-pulse motivation, BB1/CORPSE/rCinBB, SAFE-GRAPE as simultaneous amplitude-and-frequency-error correction, discretized search, loss over (ε,f), rCinBB init, tradeoff
- **UI:** Toggle rCinBB vs SAFE-GRAPE, pulse timings/phases, infidelity heatmaps, Bloch trajectories
- **Method:** Cached heatmap from repo or simplified educational version
- **Label:** Exact paper reproduction (if from SAFE_GRAPE.ipynb) / Cached / Approximate educational

### SECTION 4 — Dynamical Decoupling and Filter-Function Viewpoint
- **Question:** How does robust decoupling work?
- **Content:** Robust π-pulse for ensemble decoupling, filter function formalism, sequence A vs B, off-resonance robustness
- **UI:** Filter function vs frequency, effect of N, effect of f, A vs B comparison
- **Method:** Cached from Filter_Function_I or schematic
- **Label:** Cached / Educational reconstruction

### SECTION 5 — Thermal Feasibility and T2 Tradeoff
- **Question:** What is the heating vs coherence tradeoff?
- **Content:** Active/passive heat load, duty cycle, thermal feedback into T2/T1, optimal N
- **UI:** t_dds, N selectors, A vs B, temperature rise, T2, enhancement ratio
- **Method:** Load CSV, wrap Decoherence_sim / T2_heatmaps logic
- **Label:** Cached from Filter_Function_II CSV

### SECTION 6 — Entanglement Generation and Link Statistics
- **Question:** How do entanglement links and n_links work?
- **Content:** Single-photon protocol, geometric waiting-time, fidelity lower bound, link statistics, ε_eff, n_links
- **UI:** t_dds, t_cmpl, N, alpha, eta (if feasible), n_links and ε_eff plots, regimes where bang-bang fails
- **Method:** Cached from n_links_plots, e_eff_plots
- **Label:** Cached paper result

### SECTION 7 — Programmability and Compilation
- **Question:** How does the compiler turn entanglement into graph resources?
- **Content:** Strain-to-optical mapping, invertibility, time-bin scheduling, triangular-wave mapping, overlap-based scheduling, unique-link maximization
- **UI:** Toy interactive example (few qubits): timing, overlaps, repeated vs unique links, graph structure
- **Method:** Wrap Alg_II_heatmap, Alg_II_linkstat, strain_window_alg1; add pedagogical toy
- **Label:** Exact / Educational reconstruction

### SECTION 8 — Figure Explorer
- **Question:** Where did each figure come from?
- **UI:** Dropdown → figure number/section → title, scientific question, repo file, live/cached/approx, render, key takeaway
- **Method:** README mapping as source of truth
- **Label:** Per-figure provenance

### SECTION 9 — Reproducibility / Provenance
- **Question:** What can be reproduced and how?
- **Content:** Table: figure, reproducible?, notes, runtime, dependencies
- **Method:** Honest, explicit
- **Label:** N/A

### SECTION 10 — Limitations and Future Directions
- **Question:** What does the paper establish and where are the assumptions?
- **Content:** Summary, assumptions, simulated vs experimental, future validation
- **Method:** Careful, no overselling
- **Label:** N/A

---

## 4. Implementation Strategy

1. **Marimo App format:** Use `app = marimo.App()` and `@app.cell`; avoid accessing UI `.value` in the same cell that creates it.
2. **Figure wrappers:** Each figure module returns `plt.Figure`; wrap `tight_layout` in try/except for robustness.
3. **Cached data:** Use `app/data_loader.py` for CSV; provide "fast mode" (cached) and "full mode" (run Filter_Function_II) where practical.
4. **Labels:** Every plot cell includes a small badge: `[Exact reproduction]` / `[Cached]` / `[Educational reconstruction]`.
5. **Section template:** Each section has: Question, Method, Result, Why it matters.
6. **Modular cells:** Small cells; refactor logic into `app/figures/` and `app/content/`.

---

## 5. Assumptions and Unresolved Items

| Item | Assumption |
|------|------------|
| Authors | Not in README; will use placeholder or extract from PDF if possible |
| R1–R5 | Paper-specific; will infer from context or mark as "see paper" |
| Algorithm 1, 2 | strain_window_alg1, Alg_II_* map to these; will verify |
| LaTeX source | Not in repo; using README + PDF as source |
| filter_functions | Third-party; Filter_Function_II depends on it |
| qutip | Required for strain_window_alg1, StrainDriving |
| PyTorch | Required for SAFE_GRAPE; no live run in app |

---

## 6. Next Steps

1. Implement `paper_explorer.py` with all 10 sections.
2. Extend `app/figures/` with any missing wrappers.
3. Add `app/content/` if section prose becomes large.
4. Update `requirements.txt` and `README_EXPLORER.md`.
5. Test locally with `marimo run paper_explorer.py`.
