"""
Programmable Quantum Matter: Interactive Paper Explorer
======================================================
PRX Quantum  -  Interactive companion to:
"Programmable Quantum Matter: Heralding Large Cluster States in Driven Inhomogeneous Spin Ensembles"

Structure follows the LaTeX paper (main.tex):
  I. Introduction
  II. Theoretical framework (System description, Error-Correcting Pulses, Dynamical Decoupling)
  III. Application to Quantum Computation (Entanglement, Programmability, Strain window)
  IV. Interactive figures, Figure Explorer, Reproducibility, Limitations

Run: marimo edit paper_explorer.py  (development)
     marimo run paper_explorer.py   (app mode)
"""

import marimo
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = marimo.App(css_file=str(REPO_ROOT / "custom.css"))


# =============================================================================
# SETUP
# =============================================================================

@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from app.data_loader import csvs_available
    from app.figures.latex_figs import pdf_if_exists
    from app.figures.display_utils import fig_to_html
    return mo, plt, csvs_available, pdf_if_exists, fig_to_html


# =============================================================================
# SECTION 0  -  TITLE & AUTHORS (from main.tex)
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    from app.logos import mit_logo_data_uri, eth_logo_data_uri
    _logos = mo.hstack([
        mo.image(src=mit_logo_data_uri(), alt="MIT", height=28, style={"objectFit": "contain"}),
        mo.image(src=eth_logo_data_uri(), alt="ETH Zurich", height=25, style={"objectFit": "contain"}),
    ], justify="start", gap=2)
    mo.output.replace(mo.vstack([
        _logos,
        mo.md(r"""
# Programmable Quantum Matter
## Heralding Large Cluster States in Driven Inhomogeneous Spin Ensembles (Interactive Notebook)

**Authors:** Pratyush Anand, Louis Follet, Odiel Hooybergs, Dirk R. Englund  
*Massachusetts Institute of Technology (PA, LF, DE); ETH Zürich (OH)*

---
"""),
    ], gap=1))


# =============================================================================
# SECTION I  -  INTRODUCTION (Sec. 1 of main.tex)
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
## I. Introduction

Diamond color centers - in particular the negatively charged silicon-vacancy (SiV$^-$) center - provide high-coherence spin-photon interfaces and support optically heralded remote entanglement [<a href="#ref-1" class="citation-link">1</a>, <a href="#ref-2" class="citation-link">2</a>, <a href="#ref-3" class="citation-link">3</a>, <a href="#ref-4" class="citation-link">4</a>, <a href="#ref-5" class="citation-link">5</a>, <a href="#ref-6" class="citation-link">6</a>, <a href="#ref-7" class="citation-link">7</a>]. Building on quantum network milestones and modular frameworks for photon-mediated entanglement, these platforms offer a path toward programmable quantum network nodes [<a href="#ref-8" class="citation-link">8</a>, <a href="#ref-9" class="citation-link">9</a>, <a href="#ref-10" class="citation-link">10</a>, <a href="#ref-11" class="citation-link">11</a>, <a href="#ref-12" class="citation-link">12</a>, <a href="#ref-13" class="citation-link">13</a>, <a href="#ref-14" class="citation-link">14</a>]. Scaling such nodes imposes several requirements: spin coherence must be preserved throughout repeated probabilistic optical attempts (R1); control must be uniform and high fidelity, tolerating amplitude and detuning variations with minimal per-emitter calibration (R2) [<a href="#ref-15" class="citation-link">15</a>]; optically heralded entanglement must produce indistinguishable photons while protecting the memory state (R3) [<a href="#ref-16" class="citation-link">16</a>, <a href="#ref-17" class="citation-link">17</a>, <a href="#ref-18" class="citation-link">18</a>, <a href="#ref-19" class="citation-link">19</a>]; compilation and scheduling must orchestrate many parallel entanglement attempts with guarantees on link count and uniqueness (R4) [<a href="#ref-20" class="citation-link">20</a>, <a href="#ref-21" class="citation-link">21</a>, <a href="#ref-22" class="citation-link">22</a>, <a href="#ref-23" class="citation-link">23</a>, <a href="#ref-24" class="citation-link">24</a>, <a href="#ref-25" class="citation-link">25</a>]; and all of the above must be thermally feasible within cryogenic power budgets (R5) [<a href="#ref-26" class="citation-link">26</a>].

Today's implementations fall short on several fronts. Standard bang-bang (CPMG/XY) decoupling [<a href="#ref-27" class="citation-link">27</a>] is detuning-sensitive and relies on high peak power, leading to rapid performance loss and heating in inhomogeneous ensembles. Per-emitter interleaving of controls scales as $O(N_q)$, compressing interpulse spacings below thermalization times and eroding $T_2$ and $T_2^*$ [<a href="#ref-7" class="citation-link">7</a>, <a href="#ref-28" class="citation-link">28</a>]. Entanglement scheduling via optical switching introduces insertion loss and synchronization complexity that suppress heralding rates.

This paper replaces lossy optical-switch scheduling with **resonant, composite drive engineering**. A single optimized waveform provides global unitary control and dynamical decoupling across heterogeneous emitters - reducing control overhead from $O(N_q)$ to $O(1)$ - while maintaining high fidelity under detuning spread and lowering thermal load. Combined with a deterministic entanglement compiler, this enables optically heralded links with orders-of-magnitude higher throughput and guaranteed coverage. The framework is illustrated on SiV$^-$ centers in diamond [<a href="#ref-29" class="citation-link">29</a>, <a href="#ref-30" class="citation-link">30</a>, <a href="#ref-31" class="citation-link">31</a>, <a href="#ref-32" class="citation-link">32</a>, <a href="#ref-33" class="citation-link">33</a>, <a href="#ref-34" class="citation-link">34</a>, <a href="#ref-35" class="citation-link">35</a>], which are optically active [<a href="#ref-36" class="citation-link">36</a>, <a href="#ref-37" class="citation-link">37</a>, <a href="#ref-38" class="citation-link">38</a>], feature addressable electronic spin states (control qubits) [<a href="#ref-39" class="citation-link">39</a>, <a href="#ref-40" class="citation-link">40</a>] and nuclear spin states (memory qubits) [<a href="#ref-41" class="citation-link">41</a>, <a href="#ref-42" class="citation-link">42</a>], and have been demonstrated as quantum network nodes [<a href="#ref-43" class="citation-link">43</a>, <a href="#ref-44" class="citation-link">44</a>, <a href="#ref-45" class="citation-link">45</a>], single-photon sources [<a href="#ref-46" class="citation-link">46</a>, <a href="#ref-47" class="citation-link">47</a>], and proposed for blind quantum computing [<a href="#ref-48" class="citation-link">48</a>].
"""))


# =============================================================================
# MAIN STATIC ARCHITECTURE FIGURES (Fig. 1)
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## Architecture Overview  -  Main Figures

Having motivated the need for scalable, robust control of heterogeneous spin ensembles, we now introduce the proposed architecture. The approach uses two spatially separated spin ensembles (devices $\mathcal{M}_A$ and $\mathcal{M}_B$, with $N_a$ and $N_b$ qubits respectively) to create a bipartite cluster state - a resource for measurement-based quantum computing [<a href="#ref-50" class="citation-link">50</a>, <a href="#ref-51" class="citation-link">51</a>]. Each ensemble houses heterogeneous color centers with non-uniform spin frequencies and spin-strain susceptibilities, operating in noisy environments.

The key architectural innovation is that each ensemble is controlled by a **single global strain drive**, which provides programmable qubit addressing and is operated with optimized dynamical decoupling sequences (DDS). This reduces control overhead from $O(N_q)$ to $O(1)$. A single-photon entanglement protocol generates heralded Bell pairs between the two ensembles; these pairs are compiled into a two-dimensional bipartite-graph cluster state [<a href="#ref-52" class="citation-link">52</a>]. The figure below shows the device overview and the quantum circuit diagram of the full protocol.
"""))


@app.cell
def _(mo, pdf_if_exists):
    _overview = pdf_if_exists("overview", height="1100px")
    if _overview is not None:
        _out = mo.vstack([
            mo.md(r"**Fig. 1:** Architecture and protocol for mechanical bipartite graph cluster-state generation. **(a)** Device overview: Systems \(\mathcal{M}_{A/B}\) with heterogeneous color centers, global strain drive, DDS, single-photon entanglement, compilation. **(b)** Quantum circuit: initialization \(U_{\alpha,\phi}\), DDS channel \(\mathcal{E}_{\mathrm{DDS}}\), compilation \(\mathcal{E}_{\mathrm{cmpl}}\). [From paper]"),
            _overview,
        ])
    else:
        _out = mo.md("*Fig. 1 (overview) PDF not found in `PQM_latex_file/figs/`. Run LaTeX to generate.*")
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### Optimization objective

This architecture transforms the challenge from simple entanglement generation into a multi-parameter optimization problem. The central task is minimizing the global cost function $J(\theta)$, defined as the average two-qubit error over all generated links:

$$
J(\theta) = \left\langle \epsilon_{jk}(\theta) \right\rangle_{j,k}
$$

Here $\epsilon_{jk}$ is the error for the entanglement link between qubit $j$ (system A) and qubit $k$ (system B). Optimization is performed over the vector of experimental parameters $\theta = \{ P_{\mathrm{pulse}}, P_{\mathrm{DDS}}, P_{\mathrm{comp}} \}$, which includes: (i) the temporal shape of control pulses $P_{\mathrm{pulse}}$, (ii) the architecture of the dynamical decoupling sequences $P_{\mathrm{DDS}}$ [<a href="#ref-27" class="citation-link">27</a>], and (iii) the scheduling of entanglement attempts $P_{\mathrm{comp}}$.

The error $\epsilon_{jk}$ is a function of both coherent control errors and decoherence during the entanglement protocol (see Eq. 27 in the main text). Crucially, the fidelity decoherence for each link is determined by the compilation strategy $P_{\mathrm{comp}}$, which demands an efficient compilation algorithm to build a graph state in PSPACE [<a href="#ref-53" class="citation-link">53</a>] and polynomial time. The value of $J(\theta)$ is therefore determined by the interplay between single-qubit gate fidelities, thermally limited coherence times, and the probabilistic efficiency of the underlying single-photon entanglement protocol [<a href="#ref-16" class="citation-link">16</a>].
"""))


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### Requirements R1-R5

The paper frames scaling requirements as five conditions that a scalable quantum network node must satisfy:

| Requirement | Description |
|-------------|-------------|
| **R1** | Spin coherence preserved throughout probabilistic optical attempts; dynamical decoupling effective across heterogeneous ensembles |
| **R2** | Uniform, high-fidelity control; gates tolerate amplitude and detuning variations with minimal per-emitter calibration |
| **R3** | Optically heralded entanglement produces indistinguishable photons while protecting the memory state |
| **R4** | Compilation/scheduling orchestrates parallel entanglement attempts with guarantees on link count and uniqueness |
| **R5** | Thermally feasible within cryogenic power budgets during extended control and decoupling windows |

The integrated approach delivers four key components: (i) high-fidelity global unitary control (R2, R5), (ii) broadband dynamical decoupling (R1, R2), (iii) a single-photon remote entanglement protocol (R3), and (iv) an efficient compilation algorithm for bipartite cluster state generation (R4) [<a href="#ref-52" class="citation-link">52</a>]. The central result is that replacing lossy optical-switch scheduling with **resonant, composite drive engineering** reduces control overhead from $O(N_q)$ to $O(1)$ while maintaining high fidelity under detuning spread and lowering thermal load.
"""))


# =============================================================================
# SECTION II  -  THEORETICAL FRAMEWORK (Sec. 2 of main.tex)
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## II. Theoretical Framework

The following sections establish the theoretical foundation for the architecture. We first describe the physical system and strain-mediated control, then address error correction for inhomogeneous ensembles, and finally analyze dynamical decoupling in the frequency domain.

### II.1 System Description

Consider a set of $N_q$ group-IV color centers in diamond. The protocol is generally applicable to group-IV centers due to their shared $D_{3d}$ symmetry; we illustrate it using the transverse-oriented SiV$^-$ center. The explicit Hamiltonian is detailed in the Supplementary Information [<a href="#ref-54" class="citation-link">54</a>]. The ground-state (GS) and excited-state (ES) manifolds of each center $i$ are described by a Hamiltonian that includes strain, spin-orbit, and Zeeman interactions:

$$
\hat{\mathcal{H}}_i^{\mathrm{GS/ES}} = \hat{\mathcal{H}}_i^{\mathrm{strain}} + \hat{\mathcal{H}}_i^{\mathrm{SO(GS/ES)}} + \hat{\mathcal{H}}_i^{\mathrm{Zeeman}}
$$

We focus on the GS manifold and use its bottom two energy levels as a qubit with energy spacing $\hbar\omega_i$. An arbitrary single-qubit gate $\hat{\mathcal{U}}^{\mathrm{ideal}}(\theta, \phi) = \exp[-i\frac{\theta}{2}(\cos\phi\,\hat{\sigma}_x + \sin\phi\,\hat{\sigma}_y)]$ can be implemented using strain driving [<a href="#ref-41" class="citation-link">41</a>].

Under a **global** strain drive - where the same strain pulse acts on all $N_q$ centers - the drive frequency is detuned differently from each center: $\omega_d = \omega_i + \Delta_i$ due to different local strain environments $\epsilon_{E_{gx},i}^{\mathrm{dc}}$. Furthermore, the effective Rabi frequency varies: $\Omega_i = \Omega(1 + \epsilon_i)$. These variations encompass different strain modulation amplitudes and spin-strain susceptibilities. The real evolution unitary for center $i$ under amplitude error $\epsilon_i$ and off-resonance error $f_i = \Delta_i/\Omega$ is:

$$
\hat{\mathcal{U}}_{\epsilon_i, f_i}^{\mathrm{real}}(\theta, \phi) \cong \exp\left(-i\frac{\theta}{2}\left[(1+\epsilon_i)(\cos\phi\,\hat{\sigma}_{x,i} + \sin\phi\,\hat{\sigma}_{y,i}) - i f_i \hat{\sigma}_{z,i}\right]\right)
$$

where $\cong$ denotes the corresponding action on the bottom two energy levels. The total system Hamiltonian is:

$$
\hat{\mathcal{H}}^{\mathrm{total}} = \sum_{i=1}^{N_q} \left( \hat{\mathcal{H}}_i^{\mathrm{GS}} + \hat{\mathcal{H}}_i^{\mathrm{drive}} \right)
$$

The control problem is therefore to correct the systematic errors $\{\epsilon_i, f_i\}$ across the ensemble so that a single global drive achieves high-fidelity gates on all qubits.
"""))


# =============================================================================
# II.2 Error-Correcting Pulses (SAFE-GRAPE)
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### II.2 Error-Correcting Pulses (SAFE-GRAPE)

Composite pulses correct control errors by concatenating several primitive rotations: $\hat{\mathcal{U}}_{\mathrm{CP}}(\theta, \phi) = \prod_{i=1}^{N_p} \hat{\mathcal{U}}(\theta_i, \phi_i)$. BB1 [<a href="#ref-55" class="citation-link">55</a>] corrects amplitude errors; CORPSE [<a href="#ref-56" class="citation-link">56</a>] corrects detuning errors. **rCinBB** [<a href="#ref-57" class="citation-link">57</a>] concatenates both to address amplitude and detuning simultaneously. **SAFE-GRAPE** (Simultaneous Amplitude and Frequency Error-correcting GRadient Ascent Pulse Engineering) builds on GRAPE [<a href="#ref-58" class="citation-link">58</a>] and robust control methods [<a href="#ref-59" class="citation-link">59</a>] by optimizing over a discretized search space $\boldsymbol{\Omega}^*$ of $(\epsilon, f)$ values to minimize the weighted average infidelity:

$$
\mathcal{L} = \sum_{(\epsilon, f) \in \boldsymbol{\Omega}^*} \left[ 1 - \frac{1}{2}\mathrm{Tr}\left[ \hat{\mathcal{U}}^{\mathrm{ideal}\dagger} \prod_{i=1}^{N_p} \hat{\mathcal{U}}_{\epsilon,f}^{\mathrm{real}}(\theta_i, \phi_i) \right] \right] \mathcal{W}(\epsilon,f)
$$

Here $\mathcal{W}(\epsilon,f)$ is a weight function over the error space, and the trace term is the gate fidelity between the ideal target and the real composite unitary. SAFE-GRAPE optimizes the segment angles $\{\theta_i, \phi_i\}$ to minimize $\mathcal{L}$, yielding pulses robust across the ensemble.

**Result:** Single-qubit gate fidelities exceed 99.99% for $(\epsilon, f)$ up to 0.3, and exceed 99% for 0.4. The interactive figure below lets you explore how infidelity varies with $\epsilon$ and $f$ for both rCinBB and SAFE-GRAPE.
"""))


# ---- II.2 Interactive Fig. 2/3: Infidelity heatmap + Bloch sphere (replaces static PDF) ----
@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
**Interactive Fig. 2 (SAFE-GRAPE)**  -  Adjust $\epsilon$ (amplitude error) and $f$ (detuning) using the sliders below. The heatmaps show infidelity for rCinBB and SAFE-GRAPE; the Bloch spheres show the final state after the composite pulse. SAFE-GRAPE maintains high fidelity over a wider region of $(\epsilon, f)$. Source: `SAFE_GRAPE.ipynb`, `BandwidthAware_SAFEGRAPE.ipynb`.
"""))


@app.cell
def _(mo):
    fig2_eps = mo.ui.slider(-0.3, 0.3, value=0.25, step=0.05, label=r"$\epsilon$ (amplitude error)")
    fig2_f = mo.ui.slider(-0.3, 0.3, value=0.25, step=0.05, label=r"$f$ (detuning)")
    fig2_n_grid = mo.ui.slider(12, 25, value=15, step=1, label="Heatmap grid size")
    return (fig2_eps, fig2_f, fig2_n_grid)


@app.cell
def _(fig2_eps, fig2_f, fig2_n_grid, fig_to_html, mo):
    from app.figures.safe_grape import (
        fig2_infidelity_heatmap,
        fig2_bloch,
        fig2_fidelity_at_point,
        _safe_grape_available,
    )
    _eps = fig2_eps.value
    _f = fig2_f.value
    _n = int(fig2_n_grid.value)

    # rCinBB (always available)
    _cinbb_heatmap = fig_to_html(fig2_infidelity_heatmap(eps_range=0.3, f_range=0.3, n_grid=_n, sequence="rCinBB"))
    _cinbb_bloch = fig_to_html(fig2_bloch(epsilon=_eps, f=_f, sequence="rCinBB"))
    _cinbb_fid = fig2_fidelity_at_point(_eps, _f, sequence="rCinBB")

    # SAFE-GRAPE (requires cache)
    if _safe_grape_available():
        _grape_heatmap = fig_to_html(fig2_infidelity_heatmap(eps_range=0.3, f_range=0.3, n_grid=_n, sequence="SAFE-GRAPE"))
        _grape_bloch = fig_to_html(fig2_bloch(epsilon=_eps, f=_f, sequence="SAFE-GRAPE"))
        _grape_fid = fig2_fidelity_at_point(_eps, _f, sequence="SAFE-GRAPE")
        _grape_row = mo.hstack([_grape_heatmap, _grape_bloch], justify="center", gap=2)
        _fid_line = mo.md(rf"**Fidelity at** \((\epsilon, f) = ({_eps:.2f}, {_f:.2f})\): rCinBB = **{_cinbb_fid:.4f}**, SAFE-GRAPE = **{_grape_fid:.4f}**")
    else:
        _grape_row = mo.callout(mo.md("**SAFE-GRAPE:** Run `python scripts/generate_safe_grape_cache.py` to enable."), kind="warn")
        _fid_line = mo.md(rf"**Fidelity at** \((\epsilon, f) = ({_eps:.2f}, {_f:.2f})\): rCinBB = **{_cinbb_fid:.4f}**")

    mo.vstack([
        mo.hstack([fig2_eps, fig2_f, fig2_n_grid], justify="start", gap=2),
        _fid_line,
        mo.md("**rCinBB**"),
        mo.hstack([_cinbb_heatmap, _cinbb_bloch], justify="center", gap=2),
        mo.md("**SAFE-GRAPE**"),
        _grape_row,
    ])


# =============================================================================
# II.3 Dynamical Decoupling
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### II.3 Dynamical Decoupling: Filter-Function View

Environmental noise $\hat{\mathcal{H}}_i^{\mathrm{noise}}$ causes dephasing. For a pair of qubits $i,j$, the coherence decays as $W_{ij}(\tau) = e^{-\chi_{ij}(\tau)}$, where $\chi_{ij}$ is the decoherence functional. The filter-function formalism [<a href="#ref-60" class="citation-link">60</a>, <a href="#ref-61" class="citation-link">61</a>, <a href="#ref-62" class="citation-link">62</a>] relates $\chi_{ij}$ to the noise power spectrum $\mathcal{S}_i(\omega)$:

$$
\chi_{ij}(\tau) = \frac{2}{\pi}\int_0^\infty \frac{\mathcal{S}_i(\omega)}{\omega^2} F_{ij}(\omega\tau)\,d\omega
$$

The filter function $F_{ij}(\omega\tau)$ encodes how the DD pulse sequence responds to noise at frequency $\omega$; noise at frequencies where $F_{ij}$ is small is suppressed. The effective coherence time satisfies $\chi_{ij} \equiv (\tau/T_{2,ij})^{z_{ij}}$, where $z_{ij}$ depends on the noise spectrum (e.g., $z=2$ for $1/f$ noise).

The paper compares two CPMG-style sequences: **$\mathcal{A}_N$** uses SAFE-GRAPE-optimized $\pi$-pulses (robust to detuning), while **$\mathcal{B}_N$** uses bang-bang (instantaneous) $\pi$-pulses. CPMG builds on the Carr-Purcell [<a href="#ref-63" class="citation-link">63</a>] and Meiboom-Gill [<a href="#ref-64" class="citation-link">64</a>] sequences. Sequence $\mathcal{A}$ is approximately 30× more robust to detuning $f$ because its pulses remain effective when qubits are off-resonance. The interactive figure below lets you vary $N$ (CPMG cycles) and $f$ to compare the two sequences.
"""))


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
**Interactive Fig. 3**  -  Adjust $N$ (CPMG cycles) and $f$ (detuning) below. The filter function and coherence ratio compare sequence $\mathcal{A}$ (SAFE-GRAPE) vs $\mathcal{B}$ (bang-bang). At nonzero $f$, $\mathcal{A}$ maintains performance while $\mathcal{B}$ degrades. Source: `Filter_Function_I.ipynb`.
"""))


@app.cell
def _(mo):
    fig3_N = mo.ui.slider(1, 16, value=2, step=1, label=r"\(N\) (CPMG cycles)")
    fig3_f = mo.ui.slider(0.0, 0.8, value=0.0, step=0.1, label=r"\(f\) (detuning)")
    return (fig3_N, fig3_f)


@app.cell
def _(fig3_N, fig3_f, fig_to_html, mo):
    from app.figures.filter_function import fig3_filter_function
    n_val = int(fig3_N.value)
    f_val = float(fig3_f.value)
    try:
        _fig3 = fig3_filter_function(N=n_val, f=f_val, eps=0.0, tau=5e-3)
        _plot = fig_to_html(_fig3)
    except Exception as e:
        _plot = mo.callout(mo.md(f"**Plot error:** {e}"), kind="warn")
    mo.vstack([
        mo.hstack([fig3_N, fig3_f], justify="start", gap=2),
        _plot,
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### Thermal feedback: $T_2$ vs. temperature

The DD sequences that improve coherence also dissipate power. The resulting temperature rise $\Theta_{\mathrm{SiV}}$ at the SiV layer feeds back into the coherence time. The paper models this coupling as:

$$
T_2(\Theta_{\mathrm{SiV}}) = \frac{T_2(0.1\,\mathrm{K})}{1 + T_2(0.1\,\mathrm{K})\cdot(\Theta_{\mathrm{SiV}}-0.1)\cdot 3\times 10^6}
$$

where $T_2(0.1\,\mathrm{K})$ is the coherence time at the base temperature (0.1 K), and the denominator captures the degradation from heating. The factor $3\times 10^6$ (in appropriate units) encodes the sensitivity of $T_2$ to temperature in the SiV system.

Two competing effects determine the optimal operating point: (a) heat load from longer or more intense DD sequences raises $\Theta_{\mathrm{SiV}}$ and thus reduces $T_2$; (b) more decoupling cycles $N$ improve coherence by better filtering noise. An optimal CPMG index $N$ therefore exists that balances these effects. The interactive figure below lets you explore $T_2$ vs. $N$ and the $T_{2\mathcal{A}}/T_{2\mathcal{B}}$ ratio as a function of measurement time and sequence choice.
"""))


# ---- II.3 Interactive Fig. 4: T₂ vs N, T₂ ratio heatmap ----
@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
**Interactive Fig. 4**  -  Adjust $t_{\mathrm{dds}}$ (measurement time) and $N$ (CPMG cycles for the heatmap). Left: $T_2$ vs $N$ for sequences $\mathcal{A}$ and $\mathcal{B}$; the optimal $N$ balances decoupling benefit against thermal load. Right: $T_{2\mathcal{A}}/T_{2\mathcal{B}}$ ratio heatmap. Source: `Decoherence_sim.py`, `T2_heatmaps.py`. Requires CSV from `Filter_Function_II.py`.
"""))


@app.cell
def _(mo):
    t2_t_meas = mo.ui.slider(1e-4, 1e-3, value=2e-4, step=5e-5, label=r"$t_{\mathrm{dds}}$ (s)")
    t2_N_sel = mo.ui.dropdown(
        options={f"N={n}": n for n in [2, 4, 8, 16]},
        value="N=4",
        label="N (heatmap)",
    )
    return (t2_t_meas, t2_N_sel)


@app.cell
def _(csvs_available, t2_t_meas, t2_N_sel, fig_to_html, mo):
    _t = float(t2_t_meas.value)
    _v = t2_N_sel.value
    _n = int(_v) if isinstance(_v, (int, float)) else int(str(_v).replace("N=", ""))
    if csvs_available():
        from app.figures.decoherence import fig4_t2_vs_N as _f1, fig4_t2_heatmap as _f2
        _fig1 = _f1(t_meas=_t)
        _fig2 = _f2(N=_n, t_meas=_t)
        _out = mo.vstack([
            mo.hstack([t2_t_meas, t2_N_sel], justify="start", gap=2),
            mo.md(f"**Fig. 4(a-f):** \\(T_2\\) vs \\(N\\) at \\(t_{{\\mathrm{{dds}}}} = {_t*1e6:.0f}\\,\\mu\\)s"),
            fig_to_html(_fig1),
            mo.md(f"**Fig. 4(g-j):** \\(T_{{2\\mathcal{{A}}}}/T_{{2\\mathcal{{B}}}}\\) heatmap for \\(N={_n}\\)"),
            fig_to_html(_fig2),
        ])
    else:
        _out = mo.vstack([
            mo.hstack([t2_t_meas, t2_N_sel], justify="start", gap=2),
            mo.callout(mo.md("Run `Filter_Function_II.py` in `Dynamical Decoupling - Filter Function` to generate CSV files."), kind="warn"),
        ])
    _out


# =============================================================================
# SECTION III  -  APPLICATION TO QUANTUM COMPUTATION (Sec. 3 of main.tex)
# =============================================================================

@app.cell
def _(mo, pdf_if_exists):
    _fig = pdf_if_exists("Fig_6")
    if _fig is not None:
        _out = mo.vstack([mo.md(r"**Fig. 6:** Entanglement links statistics (checkerboard, graph representation). [From paper]"), _fig])
    else:
        _out = mo.md("*Fig. 6 PDF not found.*")
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## III. Application to Quantum Computation

With the theoretical framework for control and decoupling in place, we now turn to how the architecture is applied to quantum computation: entanglement generation, compilation, and the working strain window.

### III.1 Entanglement Operations and Cluster State Generation

The single-photon entanglement protocol uses an optically heralded photon to create remote Bell pairs between the two ensembles. The initial state of the optical mode before the beamsplitter measurement is:

$$
|\psi_{\mathrm{in}}(\alpha, \phi)\rangle = \sqrt{\alpha}|0\rangle + \sqrt{1-\alpha}\,e^{i\phi}|1\rangle
$$

where $\alpha$ is the vacuum component and $\phi$ is a phase. Upon heralding, a Bell pair is established between a qubit in system A and a qubit in system B. The protocol is probabilistic; multiple attempts are required, and the compilation strategy determines which qubits are entangled in parallel.

The figures of merit are: (i) the **effective two-qubit error** $\epsilon_{\mathrm{eff}}$, which aggregates errors across all links; and (ii) the **number of links** $n_{\mathrm{links}}$ with average error $\overline{\epsilon_{jk}} < 0.5$ (below the separable threshold). A lower bound on the average entanglement fidelity $\overline{\mathcal{F}_{jk}}$ is given in SI Eq. III.E:

$$
\overline{\mathcal{F}_{jk}} \ge \left(1 - e^{-\tau/T_1}\alpha - (1-e^{-\tau/T_1})p_{\mathrm{th}}\right) \cdot \frac{1 + e^{-((\tau/T_{2j})^{z_j}+(\tau/T_{2k})^{z_k})} - \sqrt{1-e^{-2(\tau/T_{2j})^{z_j}}}\sqrt{1-e^{-2(\tau/T_{2k})^{z_k}}}}{2}
$$

Here $\tau$ is the protocol duration, $T_1$ is the excited-state lifetime, $p_{\mathrm{th}}$ is the thermal population, and $T_{2j}$, $z_j$ are the coherence parameters for qubit $j$. The first factor captures optical and memory errors; the second captures dephasing during the protocol.

The **quantum volume** $V^i_\tau$ provides a hardware-agnostic metric [<a href="#ref-65" class="citation-link">65</a>]: $\log_2(V^i_\tau) \le \lfloor 1/\sqrt{\epsilon_{\mathrm{eff}}} \rfloor$. Lower $\epsilon_{\mathrm{eff}}$ yields higher achievable quantum volume.
"""))


# ---- III.1 Interactive: n_links, ε_eff (Fig. 6) ----
@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
**Interactive Fig. 6**  -  Adjust $t_{\mathrm{meas}}$ (measurement time) and $t_{\mathrm{cmpl}}$ (compilation time). Left: number of links $n_{\mathrm{links}}$ with $\overline{\epsilon_{jk}} < 0.5$ vs ensemble size. Right: effective two-qubit error $\epsilon_{\mathrm{eff}}$ vs ensemble size. Longer $t_{\mathrm{meas}}$ allows more decoupling; shorter $t_{\mathrm{cmpl}}$ enables denser scheduling. Cached from `n_links_plots.py`, `e_eff_plots.py`.
"""))


@app.cell
def _(mo):
    _t_meas_opts = {"200 µs": 2e-4, "800 µs": 8e-4}
    _t_cmpl_opts = {"1 µs": 1e-6, "10 µs": 1e-5}
    fig6_t_meas = mo.ui.dropdown(
        options=_t_meas_opts,
        value="800 µs",
        label=r"$t_{\mathrm{meas}}$",
    )
    fig6_t_cmpl = mo.ui.dropdown(
        options=_t_cmpl_opts,
        value="10 µs",
        label=r"$t_{\mathrm{cmpl}}$",
    )
    return (fig6_t_meas, fig6_t_cmpl)


@app.cell
def _(fig6_t_meas, fig6_t_cmpl, fig_to_html, mo):
    from app.figures.entanglement import fig6_n_links_interactive as _f_links, fig6_epsilon_eff_interactive as _f_eff
    _t_meas = float(fig6_t_meas.value) if fig6_t_meas.value is not None else 8e-4
    _t_cmpl = float(fig6_t_cmpl.value) if fig6_t_cmpl.value is not None else 1e-5
    _links_fig = _f_links(t_meas=_t_meas, t_cmpl=_t_cmpl)
    _eff_fig = _f_eff(t_meas=_t_meas, t_cmpl=_t_cmpl)
    mo.vstack([
        mo.hstack([fig6_t_meas, fig6_t_cmpl], justify="start", gap=2),
        mo.hstack([
            mo.vstack([mo.md("**Fig. 6(a-b): n_links**"), fig_to_html(_links_fig)]),
            mo.vstack([mo.md("**Fig. 6(c-d): ε_eff**"), fig_to_html(_eff_fig)]),
        ], justify="center", gap=2),
    ])


# =============================================================================
# III.2 Programmability and Compilation
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### III.2 Programmability and Compilation

To address multiple qubits with a single global strain drive, the strain waveform must be designed so that each qubit is brought into resonance with the optical transition at the right time. The **strain-to-optical mapping** relates the applied strain $s_a$ to the optical frequency of qubit $j$: $f_j = \mathbb{F}_{aj}(s_a)$. Under the assumption of invertibility, the strain required to address qubit $j$ at the target frequency $f_L$ is $s_{aj} = \mathbb{F}_{aj}^{-1}(f_L)$.

**Algorithm 1** computes the global drive $\tilde{s}_a(t)$ by ordering the addressing times, linearly interpolating between them, and applying palindromic or periodic extension to form a continuous waveform. The **triangular-wave** qubit sequence assigns each qubit an ordering index:

$$
\theta_{ja} = 1 + \min\left\{ j \bmod (2N_a),\;(2N_a - 1) - (j \bmod (2N_a)) \right\}
$$

This creates a deterministic, periodic schedule for which qubit is addressed at each time.

**Algorithm 2** addresses the bipartite case: given $N_a$ and $N_b$ qubits in systems A and B, it finds the scaling $m_{\mathrm{scal}} = T_A/T_B$ (ratio of cycle times) that maximizes the number of unique entanglement links. The guarantee is $\sum_{i\ge 1} h_i \ge \max(N_a, N_b)/(N_a N_b)$, where $h_i$ are link counts, yielding $\Omega(N_q)$ unique links in polynomial time. The interactive figure below shows heatmaps of link statistics and scaling as a function of $N_a$ and $N_b$.
"""))


@app.cell
def _(mo, pdf_if_exists):
    _fig = pdf_if_exists("Fig_9_new", height="1000px")
    if _fig is not None:
        _out = mo.vstack([mo.md(r"**Fig. 8:** Simulation for entanglement compilation (Algorithm II). Heatmaps (a-c) and link statistics (d). [From paper  -  `Alg_II_heatmap.py`, `Alg_II_linkstat.py`]"), _fig])
    else:
        _out = mo.md("*Fig. 8 PDF not found.*")
    return _out


# ---- III.2 Interactive: Fig. 8 (Compilation) heatmaps and linkstat ----
@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
**Interactive Fig. 8**  -  Heatmaps show link statistics (unique links, coverage) as a function of $N_a$ and $N_b$. The link-statistics panel shows the distribution for a chosen $(N_a, N_b)$. Algorithm 2 finds the scaling $m_{\mathrm{scal}}$ that maximizes unique links. Source: `Alg_II_heatmap.py`, `Alg_II_linkstat.py`.
"""))


@app.cell
def _(mo):
    _na_opts = {str(n): n for n in range(2, 11)}
    _nb_opts = {str(n): n for n in range(2, 11)}
    fig8_Na_max = mo.ui.dropdown(options=_na_opts, value="8", label=r"$N_a$ max (heatmaps)")
    fig8_Nb_max = mo.ui.dropdown(options=_nb_opts, value="8", label=r"$N_b$ max (heatmaps)")
    fig8_Na = mo.ui.dropdown(options=_na_opts, value="5", label=r"$N_a$ (link stat)")
    fig8_Nb = mo.ui.dropdown(options=_nb_opts, value="4", label=r"$N_b$ (link stat)")
    return (fig8_Na_max, fig8_Nb_max, fig8_Na, fig8_Nb)


@app.cell
def _(fig8_Na_max, fig8_Nb_max, fig8_Na, fig8_Nb, fig_to_html, mo):
    from app.figures.compilation import fig8_heatmaps as _f8, fig8_linkstat as _f8l
    _na_max = int(fig8_Na_max.value) if fig8_Na_max.value is not None else 8
    _nb_max = int(fig8_Nb_max.value) if fig8_Nb_max.value is not None else 8
    _na = int(fig8_Na.value) if fig8_Na.value is not None else 5
    _nb = int(fig8_Nb.value) if fig8_Nb.value is not None else 4
    try:
        _fig_heat = _f8(Na_max=_na_max, Nb_max=_nb_max)
        _fig_link = _f8l(Na=_na, Nb=_nb)
        mo.output.replace(mo.hstack([fig8_Na_max, fig8_Nb_max, fig8_Na, fig8_Nb], justify="start", gap=2))
        mo.output.append(mo.md("**Fig. 8(a-c):** Heatmaps"))
        mo.output.append(fig_to_html(_fig_heat, dpi=72))
        mo.output.append(mo.md("**Fig. 8(d):** Link statistics"))
        mo.output.append(fig_to_html(_fig_link, dpi=72))
    except Exception as e:
        mo.output.replace(mo.callout(mo.md(f"**Error:** {e}"), kind="danger"))


# =============================================================================
# III.3 Working Strain Window
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
### III.3 Working Strain Window (Algorithm 1)

For the strain drive to address each qubit uniquely, the total strain at each center must vary monotonically through a range that includes the target optical frequency $f_L$. The strain at center $i$ is decomposed into three contributions:

$$
\epsilon_{E_{gx},i}(t) = \underbrace{\epsilon_{E_{gx},i}^{\mathrm{bias}}}_{\mathrm{fabrication}} + \underbrace{\epsilon_{E_{gx}}^{\mathrm{dc}}}_{\mathrm{piezo}} + \underbrace{\epsilon_{E_{gx}}^{\mathrm{ac}}(t)}_{\mathrm{control}}
$$

The **bias** term is set by fabrication (e.g., built-in strain from the diamond substrate); the **dc** term is from a quasi-static piezo setting; and the **ac** term is the time-dependent control drive. In the low-strain limit, the optical transition frequency varies linearly: $f_{\mathrm{opt},i} = f_0 + \Delta d\,\epsilon_{E_{gx},i}$ with $\Delta d \simeq 0.5$ PHz/strain.

A **common monotonic strain window** exists when the combined strain $\epsilon_{E_{gx},i}(t)$ sweeps through $f_L$ exactly once for every center $i$, with no two centers crossing simultaneously. Algorithm 1 ensures this by ordering the bias strains and designing $\epsilon_{E_{gx}}^{\mathrm{ac}}(t)$ accordingly. The interactive figure below visualizes the strain window and the addressing sequence.
"""))


@app.cell
def _(mo, pdf_if_exists):
    _fig = pdf_if_exists("strain_window", height="800px")
    if _fig is not None:
        _out = mo.vstack([mo.md(r"**Fig. 9:** Common monotonic strain window. [From paper  -  `strain_window_alg1.py`]"), _fig])
    else:
        _out = mo.md("*Fig. 9 PDF not found.*")
    return _out


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
**Interactive Fig. 9**  -  Strain window for Algorithm 1. Adjust strain range, $\sigma$ (bias distribution width), and number of quantiles. The plot shows how the ac strain sweep brings each center through $f_L$ in a monotonic order. Source: `strain_window_alg1.py`. Requires qutip.
"""))


@app.cell
def _(mo):
    _strain_opts = {"6e-5": 6e-5, "9e-5": 9e-5, "12e-5": 12e-5}
    _sigma_opts = {"4e-5": 4e-5, "5e-5": 5e-5, "6e-5": 6e-5, "7e-5": 7e-5}
    _nq_opts = {str(n): n for n in range(5, 16)}
    fig9_strain_range = mo.ui.dropdown(options=_strain_opts, value="9e-5", label="Strain range")
    fig9_sigma = mo.ui.dropdown(options=_sigma_opts, value="6e-5", label=r"$\sigma$ width")
    fig9_n_quantiles = mo.ui.dropdown(options=_nq_opts, value="10", label="N quantiles")
    return (fig9_strain_range, fig9_sigma, fig9_n_quantiles)


@app.cell
def _(fig9_strain_range, fig9_sigma, fig9_n_quantiles, fig_to_html, mo):
    from app.figures.compilation import fig9_strain_window
    _sr = float(fig9_strain_range.value) if fig9_strain_range.value is not None else 9e-5
    _sw = float(fig9_sigma.value) if fig9_sigma.value is not None else 6e-5
    _nq = int(fig9_n_quantiles.value) if fig9_n_quantiles.value is not None else 10
    try:
        _fig = fig9_strain_window(strain_range=_sr, sigma_width=_sw, n_quantiles=_nq, n_points=120)
        mo.output.replace(mo.hstack([fig9_strain_range, fig9_sigma, fig9_n_quantiles], justify="start", gap=2))
        mo.output.append(fig_to_html(_fig, dpi=72))
    except Exception as e:
        mo.output.replace(mo.callout(mo.md(f"**Error:** {e}"), kind="danger"))


# =============================================================================
# SECTION IV  -  HEAT MODELING / THERMAL BUDGET (Fig. S11, S12  -  SI Sec. III.F)
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## IV. Heat Modeling (Thermal Budget  -  SI Sec. III.F)

The control and decoupling sequences dissipate power into the sample. This section models how that heat flows through the cryogenic setup and how it feeds back into coherence times - completing the loop between R1 (coherence), R2 (control), and R5 (thermal feasibility).

The system is placed on the cold-plate stage (≈100 mK) of a dilution refrigerator. Implementing a pulse sequence ($\mathcal{A}$ or $\mathcal{B}$) produces active and passive heat loads that depend on dilution fridge geometry, thermal properties of cryogenic coax, sample footprint, specific heat capacity, and thermal anchoring [<a href="#ref-26" class="citation-link">26</a>]. The cold-plate temperature $\overline{\Theta}_{\mathrm{CP}}$ accounts for slow temperature rise from attenuators and duty-cycle correction, calibrated following Krinner et al. The thermal environment of the SiV layer is modeled with fast heating/cooling dynamics: Debye heat capacity, pulse-train superposition, and a thermalization timescale $\tau_{\mathrm{th,SiV}}$.

The SiV temperature $\Theta_{\mathrm{SiV}}(t)$ is given by SI Eq. (89): it combines $\overline{\Theta}_{\mathrm{CP}}$ with a transient term $\Theta_{\mathrm{SiV}}(t,t_0)$ that captures the response to each pulse. Assuming $1/T_2$ and $1/T_1$ depend linearly on $\Theta_{\mathrm{SiV}}$ in the low-temperature regime (SI Eq. 91-92), the effective $T_2$ during a measurement window $t_{\mathrm{dds}}$ is computed for both sequences. Sequence $\mathcal{B}$ (bang-bang) delivers more heat than $\mathcal{A}$ (SAFE-GRAPE) because it uses shorter, higher-peak-power pulses; at $t_{\mathrm{dds}}=0.2$ ms, $\Theta_{\mathrm{SiV}}$ rises ~5.2% for $\mathcal{B}$ vs ~0.3% for $\mathcal{A}$. The figures below show cold-plate temperature vs CPMG index $N$ and SiV fast temperature vs time.
"""))


@app.cell
def _(mo):
    fig11_t_dds = mo.ui.dropdown(
        options={"All": None, "0.2 ms": 2e-4, "0.8 ms": 8e-4, "2 ms": 2e-3},
        value="All",
        label=r"Fig. S11: $t_{\mathrm{dds}}$",
    )
    fig12_t_dds = mo.ui.dropdown(
        options={"0.2 ms": 2e-4, "2 ms": 2e-3},
        value="0.2 ms",
        label=r"Fig. S12: $t_{\mathrm{dds}}$",
    )
    fig12_sequence = mo.ui.dropdown(
        options={"A (SAFE-GRAPE)": "A", "B (bang-bang)": "B"},
        value="A (SAFE-GRAPE)",
        label="Sequence",
    )
    return (fig11_t_dds, fig12_t_dds, fig12_sequence)


@app.cell
def _(mo, fig_to_html, fig11_t_dds, fig12_t_dds, fig12_sequence):
    from app.figures.heat_modeling import fig11_dil_fridge, fig12_siv_fast
    _t11 = fig11_t_dds.value
    _t12 = float(fig12_t_dds.value)
    _seq = fig12_sequence.value
    t_filter = None if _t11 is None else [_t11]
    _f11 = fig11_dil_fridge(t_meas_filter=t_filter)
    _f12 = fig12_siv_fast(t_meas=_t12, sequence=_seq)
    mo.vstack([
        mo.hstack([fig11_t_dds, fig12_t_dds, fig12_sequence], justify="start", gap=2),
        mo.md(r"**Fig. S11:** Cold-plate temperature $\overline{\Theta}_{CP}$ vs CPMG index $N$. Sequence $\mathcal{B}$ heats more than $\mathcal{A}$; longer $t_{\mathrm{dds}}$ reduces duty cycle."),
        fig_to_html(_f11),
        mo.md(r"**Fig. S12:** SiV fast temperature $\Theta_{\mathrm{SiV}}$ vs time. Heat accumulates when pulses arrive faster than thermalization."),
        fig_to_html(_f12),
    ])


# =============================================================================
# FIGURE EXPLORER
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## Figure Explorer

The notebook has presented the main results through interactive figures. Use the dropdown below to select any figure and view its scientific question, source file, and reproducibility instructions. This aids navigation and verification of the paper's claims.
"""))


@app.cell
def _(mo):
    figure_choice = mo.ui.dropdown(
        options={
            "Fig. 2 (SAFE-GRAPE)": "fig2",
            "Fig. 3 (Filter function)": "fig3",
            "Fig. 4 (T₂)": "fig4",
            "Fig. 6 (Entanglement)": "fig6",
            "Fig. 8 (Compilation)": "fig8",
            "Fig. 9 (Strain window)": "fig9",
            "Fig. S11 (Cold-plate)": "fig11",
            "Fig. S12 (SiV temp)": "fig12",
        },
        value="Fig. 4 (T₂)",
        label="Figure",
    )
    mo.vstack([figure_choice])
    return figure_choice,


@app.cell
def _(figure_choice, mo):
    FIGURE_INFO = {
        "fig2": ("Fig. 2", "SAFE_GRAPE.ipynb", "Pulse fidelity vs (ε, f)", "Run notebook (PyTorch)"),
        "fig3": ("Fig. 3(b-d)", "Filter_Function_I.ipynb", "Filter function vs frequency", "Run notebook"),
        "fig4": ("Fig. 4", "Decoherence_sim.py, T2_heatmaps.py", "T₂ vs N; T₂ ratio heatmaps", "Load CSV (Filter_Function_II)"),
        "fig6": ("Fig. 6(a-d)", "n_links_plots.py, e_eff_plots.py", "n_links, ε_eff vs N", "Cached paper result"),
        "fig8": ("Fig. 8", "Alg_II_heatmap.py, Alg_II_linkstat.py", "Link statistics heatmaps", "Live regeneration"),
        "fig9": ("Fig. 9", "strain_window_alg1.py", "Strain window for SiV", "Live (qutip) or schematic"),
        "fig11": ("Fig. S11", "dil_fridge.py", "Cold-plate temperature vs N", "Live regeneration"),
        "fig12": ("Fig. S12", "temp_siv_fast.py", "SiV fast temperature vs time", "Live regeneration"),
    }
    _fig_map = {"Fig. 2 (SAFE-GRAPE)": "fig2", "Fig. 3 (Filter function)": "fig3", "Fig. 4 (T₂)": "fig4",
                "Fig. 6 (Entanglement)": "fig6", "Fig. 8 (Compilation)": "fig8", "Fig. 9 (Strain window)": "fig9",
                "Fig. S11 (Cold-plate)": "fig11", "Fig. S12 (SiV temp)": "fig12"}
    info = FIGURE_INFO.get(_fig_map.get(figure_choice.value, "fig4"), FIGURE_INFO["fig4"])
    name, source, meaning, repro = info
    mo.md(f"**{name}**\n\n- **Scientific question:** {meaning}\n- **Source:** `{source}`\n- **Reproducibility:** {repro}")
    return


# =============================================================================
# REPRODUCIBILITY & LIMITATIONS
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## Reproducibility and Provenance

All figures in this notebook can be reproduced from the source code. The table below summarizes the workflow for each figure:

| Figure | Reproducible? | Notes |
|--------|---------------|-------|
| Fig. 2 | Yes (run notebook) | SAFE_GRAPE.ipynb, PyTorch |
| Fig. 3 | Yes (run notebook) | Filter_Function_I.ipynb |
| Fig. 4 | Yes (with CSV) | Run Filter_Function_II.py first to generate filter-function data |
| Fig. 6 | Yes (cached) | Precomputed from qv_vs_t_cmpl_A/B; live regeneration available |
| Fig. 8 | Yes (live) | Alg_II_heatmap.py, Alg_II_linkstat.py |
| Fig. 9 | Yes (qutip) | strain_window_alg1.py |
| Fig. S11, S12 | Yes (live) | dil_fridge.py, temp_siv_fast.py |

**CSV provenance:** Fig. 4 requires `N=2,4,8,16_results_parallel.csv` from `Filter_Function_II.py` in the Dynamical Decoupling module.

**Dependencies:** numpy, matplotlib, pandas, scipy, marimo; qutip (Fig. 9, Filter_Function_II); filter_functions (Filter_Function_II); torch (SAFE_GRAPE).
"""))


@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.callout(mo.md(r"""
---
## Limitations and Future Directions

**What the paper establishes:** An integrated architecture for programmable quantum matter in heterogeneous spin ensembles, with robust control (SAFE-GRAPE, bandwidth-aware DD), thermal feasibility within cryogenic budgets, and an efficient compilation algorithm for bipartite cluster states. Control overhead is reduced from $O(N_q)$ to $O(1)$ at the global level.

**Where assumptions enter:** The strain-to-optical mapping $\mathbb{F}_{aj}$ is assumed invertible (Algorithm 1, Fig. 9); the environmental noise spectrum $\mathcal{S}_i(\omega)$ is modeled; thermal parameters (heat capacity, thermalization times, cold-plate coupling) are taken from literature or estimated.

**Simulated vs experimentally demonstrated:** The paper presents a theoretical and numerical framework. Experimental validation of the full pipeline - from strain drive to heralded entanglement - remains future work.

**Future validation:** Hardware-constrained pulse optimization (Fig. S4), in-situ thermal characterization, and demonstration of heralded entanglement at scale.
"""), kind="warn"))


# =============================================================================
# REFERENCES
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## References

Citations in the text use numbers [<a href="#ref-1" class="citation-link">1</a>], [<a href="#ref-2" class="citation-link">2</a>], [<a href="#ref-3" class="citation-link">3</a>], … corresponding to the LaTeX paper bibliography. Click any citation to jump to its entry below. The full BibTeX file is `PQM_latex_file/bib.bib`. Each citation is listed separately:

| # | Citation |
|---|----------|
| 1 | <span id="ref-1"></span>[Beukers et al., PRX Quantum **5**, 010202 (2024)](https://link.aps.org/doi/10.1103/PRXQuantum.5.010202) |
| 2 | <span id="ref-2"></span>[Sukachev et al., Phys. Rev. Lett. **119**, 223602 (2017)](https://link.aps.org/doi/10.1103/PhysRevLett.119.223602) |
| 3 | <span id="ref-3"></span>[Arjona Martínez et al., Phys. Rev. Lett. **129**, 173603 (2022)](https://link.aps.org/doi/10.1103/PhysRevLett.129.173603) |
| 4 | <span id="ref-4"></span>[Knaut et al., Nature **629**, 573 (2024)](https://doi.org/10.1038/s41586-024-07252-z) |
| 5 | <span id="ref-5"></span>[Iwasaki et al., Phys. Rev. Lett. **119**, 253601 (2017)](https://link.aps.org/doi/10.1103/PhysRevLett.119.253601) |
| 6 | <span id="ref-6"></span>[Parker et al., Nat. Photon. **18**, 156 (2024)](https://doi.org/10.1038/s41566-023-01337-3) |
| 7 | <span id="ref-7"></span>[Rugar et al., Phys. Rev. X **11**, 031021 (2021)](https://link.aps.org/doi/10.1103/PhysRevX.11.031021) |
| 8 | <span id="ref-8"></span>[Barrett & Kok, Phys. Rev. A **71**, 060310 (2005)](https://link.aps.org/doi/10.1103/PhysRevA.71.060310) |
| 9 | <span id="ref-9"></span>[Bernien et al., Nature **497**, 86 (2013)](https://doi.org/10.1038/nature12016) |
| 10 | <span id="ref-10"></span>[Hensen et al., Nature **526**, 682 (2015)](https://doi.org/10.1038/nature15759) |
| 11 | <span id="ref-11"></span>[Humphreys et al., Nature **558**, 268 (2018)](https://doi.org/10.1038/s41586-018-0200-y) |
| 12 | <span id="ref-12"></span>[Pompili et al., Science **372**, 259 (2021)](https://www.science.org/doi/10.1126/science.abf1919) |
| 13 | <span id="ref-13"></span>[Stas et al., Science **378**, 557 (2022)](https://www.science.org/doi/10.1126/science.add9771) |
| 14 | <span id="ref-14"></span>[Bersin et al., PRX Quantum **5**, 010303 (2024)](https://link.aps.org/doi/10.1103/PRXQuantum.5.010303) |
| 15 | <span id="ref-15"></span>[Dong & Petersen, IET Control Theory Appl. **4**, 2651 (2010)](https://doi.org/10.1049/iet-cta.2009.0508) |
| 16 | <span id="ref-16"></span>[Hermans et al., New J. Phys. **25**, 013011 (2023)](https://doi.org/10.1088/1367-2630/acb004) |
| 17 | <span id="ref-17"></span>[Barrett & Kok, Phys. Rev. A **71**, 060310 (2005)](https://link.aps.org/doi/10.1103/PhysRevA.71.060310) |
| 18 | <span id="ref-18"></span>[Bernien et al., Nature **497**, 86 (2013)](https://doi.org/10.1038/nature12016) |
| 19 | <span id="ref-19"></span>[Pompili et al., Science **372**, 259 (2021)](https://www.science.org/doi/10.1126/science.abf1919) |
| 20 | <span id="ref-20"></span>[Maronese et al., arXiv:2112.00187 (2021)](https://arxiv.org/abs/2112.00187) |
| 21 | <span id="ref-21"></span>[Wang et al., ISCA 293 (2024)](https://doi.org/10.1145/3620666.3650186) |
| 22 | <span id="ref-22"></span>[Li et al., arXiv:2405.16380 (2024)](https://arxiv.org/abs/2405.16380) |
| 23 | <span id="ref-23"></span>[Patil et al., npj Quantum Inf. **8**, 51 (2022)](https://doi.org/10.1038/s41534-022-00561-5) |
| 24 | <span id="ref-24"></span>[Kaur & Guha, arXiv:2306.03319 (2023)](https://arxiv.org/abs/2306.03319) |
| 25 | <span id="ref-25"></span>[Humphreys et al., Nature **558**, 268 (2018)](https://doi.org/10.1038/s41586-018-0200-y) |
| 26 | <span id="ref-26"></span>[Krinner et al., EPJ Quantum Technol. **6**, 2 (2019)](https://doi.org/10.1140/epjqt/s40507-019-0072-0) |
| 27 | <span id="ref-27"></span>[Viola & Lloyd, Phys. Rev. A **58**, 2733 (1998)](https://link.aps.org/doi/10.1103/PhysRevA.58.2733) |
| 28 | <span id="ref-28"></span>[Nguyen et al., Phys. Rev. B **100**, 165428 (2019)](https://link.aps.org/doi/10.1103/PhysRevB.100.165428) |
| 29 | <span id="ref-29"></span>[Bradac et al., Nat. Commun. **10**, 5625 (2019)](https://doi.org/10.1038/s41467-019-13332-w) |
| 30 | <span id="ref-30"></span>[Hepp et al., Phys. Rev. Lett. **112**, 036405 (2014)](https://link.aps.org/doi/10.1103/PhysRevLett.112.036405) |
| 31 | <span id="ref-31"></span>[Iwasaki et al., Sci. Rep. **5**, 12882 (2015)](https://doi.org/10.1038/srep12882) |
| 32 | <span id="ref-32"></span>[Bhaskar et al., Phys. Rev. Lett. **118**, 223603 (2017)](https://link.aps.org/doi/10.1103/PhysRevLett.118.223603) |
| 33 | <span id="ref-33"></span>[Iwasaki et al., Phys. Rev. Lett. **119**, 253601 (2017)](https://link.aps.org/doi/10.1103/PhysRevLett.119.253601) |
| 34 | <span id="ref-34"></span>[Debroux et al., Phys. Rev. X **11**, 041041 (2021)](https://link.aps.org/doi/10.1103/PhysRevX.11.041041) |
| 35 | <span id="ref-35"></span>[Beukers et al., Phys. Rev. X **15**, 021011 (2025)](https://link.aps.org/doi/10.1103/PhysRevX.15.021011) |
| 36 | <span id="ref-36"></span>[Clark et al., Phys. Rev. B **51**, 16681 (1995)](https://link.aps.org/doi/10.1103/PhysRevB.51.16681) |
| 37 | <span id="ref-37"></span>[Goss et al., Phys. Rev. Lett. **77**, 3041 (1996)](https://link.aps.org/doi/10.1103/PhysRevLett.77.3041) |
| 38 | <span id="ref-38"></span>[Sukachev et al., Phys. Rev. Lett. **119**, 223602 (2017)](https://link.aps.org/doi/10.1103/PhysRevLett.119.223602) |
| 39 | <span id="ref-39"></span>[Pingault et al., Nat. Commun. **8**, 15579 (2017)](https://doi.org/10.1038/ncomms15579) |
| 40 | <span id="ref-40"></span>[Meesala et al., Phys. Rev. B **97**, 205444 (2018)](https://link.aps.org/doi/10.1103/PhysRevB.97.205444) |
| 41 | <span id="ref-41"></span>[Metsch et al., Phys. Rev. Lett. **122**, 190503 (2019)](https://link.aps.org/doi/10.1103/PhysRevLett.122.190503) |
| 42 | <span id="ref-42"></span>[Stas et al., Science **378**, 557 (2022)](https://www.science.org/doi/10.1126/science.add9771) |
| 43 | <span id="ref-43"></span>[Bhaskar et al., Nature **580**, 60 (2020)](https://doi.org/10.1038/s41586-020-2103-5) |
| 44 | <span id="ref-44"></span>[Stas et al., Science **378**, 557 (2022)](https://www.science.org/doi/10.1126/science.add9771) |
| 45 | <span id="ref-45"></span>[Bersin et al., PRX Quantum **5**, 010303 (2024)](https://link.aps.org/doi/10.1103/PRXQuantum.5.010303) |
| 46 | <span id="ref-46"></span>[Sipahigil et al., Phys. Rev. Lett. **113**, 113602 (2014)](https://link.aps.org/doi/10.1103/PhysRevLett.113.113602) |
| 47 | <span id="ref-47"></span>[Knall et al., Phys. Rev. Lett. **129**, 053603 (2022)](https://link.aps.org/doi/10.1103/PhysRevLett.129.053603) |
| 48 | <span id="ref-48"></span>[Wei et al., Science **388**, 509 (2025)](https://www.science.org/doi/10.1126/science.adu6894) |
| 49 | <span id="ref-49"></span>[Ni et al., arXiv:2505.12461 (2025)](https://arxiv.org/abs/2505.12461) |
| 50 | <span id="ref-50"></span>[Jozsa, arXiv:quant-ph/0508124 (2005)](https://arxiv.org/abs/quant-ph/0508124) |
| 51 | <span id="ref-51"></span>[Kok et al., Rev. Mod. Phys. **79**, 135 (2007)](https://link.aps.org/doi/10.1103/RevModPhys.79.135) |
| 52 | <span id="ref-52"></span>[Nielsen, Rep. Math. Phys. **57**, 147 (2006)](https://doi.org/10.1016/S0034-4877(06)80014-5) |
| 53 | <span id="ref-53"></span>[Jain et al., arXiv:0907.4737 (2009)](https://arxiv.org/abs/0907.4737) |
| 54 | <span id="ref-54"></span>[Supplementary Information, Programmable Quantum Matter (2025)](https://drive.google.com/drive/folders/1gqB1cdJpg_otDfSoiKS4JsFGrrLeY_Bv) |
| 55 | <span id="ref-55"></span>[Wimperis, J. Magn. Reson. A **109**, 221 (1994)](https://doi.org/10.1006/jmra.1994.1159) |
| 56 | <span id="ref-56"></span>[Cummins et al., Phys. Rev. A **67**, 042308 (2003)](https://link.aps.org/doi/10.1103/PhysRevA.67.042308) |
| 57 | <span id="ref-57"></span>[Bando et al., J. Phys. Soc. Jpn. **82**, 014004 (2013)](https://doi.org/10.7566/JPSJ.82.014004) |
| 58 | <span id="ref-58"></span>[Khaneja et al., J. Magn. Reson. **172**, 296 (2005)](https://doi.org/10.1016/j.jmr.2004.11.004) |
| 59 | <span id="ref-59"></span>[Said & Twamley, Phys. Rev. A **80**, 032303 (2009)](https://link.aps.org/doi/10.1103/PhysRevA.80.032303) |
| 60 | <span id="ref-60"></span>[Biercuk et al., J. Phys. B **44**, 154002 (2011)](https://doi.org/10.1088/0953-4075/44/15/154002) |
| 61 | <span id="ref-61"></span>[Hangleiter et al., Phys. Rev. Res. **3**, 043047 (2021)](https://link.aps.org/doi/10.1103/PhysRevResearch.3.043047) |
| 62 | <span id="ref-62"></span>[Cerfontaine et al., Phys. Rev. Lett. **127**, 170403 (2021)](https://link.aps.org/doi/10.1103/PhysRevLett.127.170403) |
| 63 | <span id="ref-63"></span>[Carr & Purcell, Phys. Rev. **94**, 630 (1954)](https://link.aps.org/doi/10.1103/PhysRev.94.630) |
| 64 | <span id="ref-64"></span>[Meiboom & Gill, Rev. Sci. Instrum. **29**, 688 (1958)](https://doi.org/10.1063/1.1716296) |
| 65 | <span id="ref-65"></span>[Cross et al., Phys. Rev. A **100**, 032328 (2019)](https://link.aps.org/doi/10.1103/PhysRevA.100.032328) |
"""))


# =============================================================================
# APPENDIX
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.output.replace(mo.md(r"""
---
## Appendix: Key Parameters

Reference table of symbols and typical values used throughout the notebook:

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $N$ | CPMG concatenation index | 2, 4, 8, 16 |
| $\epsilon$ | Amplitude error (fractional deviation of Rabi frequency) | ±0.3-0.5 |
| $f$ | Detuning / off-resonance error ($\Delta/\Omega$) | ±0.3-0.5 |
| $t_{\mathrm{dds}}$ | DDS measurement time | 0.2-0.8 ms |
| $t_{\pi}^{\mathcal{A}}$ | $\pi$ pulse duration for sequence $\mathcal{A}$ (SAFE-GRAPE) | 150 ns |
| $t_{\pi}^{\mathcal{B}}$ | $\pi$ pulse duration for sequence $\mathcal{B}$ (bang-bang) | 10 ns |
| $T_{\mathrm{CP}}$ | Cold-plate base temperature | 100 mK |
| $\Theta_{\mathrm{SiV}}$ | SiV layer temperature (includes heating) |  -  |
| $N_a$, $N_b$ | Qubit counts in systems A and B |  -  |
"""))
