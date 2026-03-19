"""
Resolve repository paths for the paper explorer app.
All original scripts assume they run from their respective folders;
this module provides absolute paths so the app can run from the repo root.
"""
from pathlib import Path

# Repo root: parent of app/ directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# LaTeX source and figures (from PQM_latex_file)
LATEX_ROOT = REPO_ROOT / "PQM_latex_file"
LATEX_FIGS = LATEX_ROOT / "figs"

# Paper figures from LaTeX (fig name -> PDF path)
# Only keys with existing files will render. Main text: Fig 1-2 (overview), Fig 3 (SAFE-GRAPE),
# Master_Fig_3 (filter), T2_main (Fig 4), Fig 6 (graph), QV (entanglement), Fig 8 (programmability),
# Fig 9 (compiler), Fig 10 (strain).
PAPER_FIGS = {
    "overview": LATEX_FIGS / "Fig_1-2_new.pdf",
    "SAFE_GRAPE": LATEX_FIGS / "Fig_3.pdf",
    "filter_function": LATEX_FIGS / "Master_Fig_3.pdf",
    "T2_main": LATEX_FIGS / "T2_main.pdf",
    "Fig_5": LATEX_FIGS / "Fig_5.pdf",
    "Fig_6": LATEX_FIGS / "Fig_6.pdf",
    "QV": LATEX_FIGS / "QV.pdf",
    "Fig_7": LATEX_FIGS / "Fig_7.pdf",
    "Fig_8": LATEX_FIGS / "Fig_8.pdf",
    "Fig_9": LATEX_FIGS / "Fig_9.pdf",
    "Fig_9_new": LATEX_FIGS / "Fig_9_new.pdf",
    "strain_window": LATEX_FIGS / "Fig_10.pdf",
    "SI_strain_driving": LATEX_FIGS / "SI_Fig_2.pdf",
    "BandwidthAware": LATEX_FIGS / "BandwidthAwareSAFEGRAPE.pdf",
    "scaling": LATEX_FIGS / "scaling.pdf",
    "T_SiV": LATEX_FIGS / "T_SiV.pdf",
}

# Section folders
ERROR_CORRECTING_PULSES = REPO_ROOT / "Error-Correcting Pulses_"
FILTER_FUNCTION = REPO_ROOT / "Dynamical Decoupling - Filter Function"
TEMPERATURE_T2 = REPO_ROOT / "Temperature and T2 simulations"
ENTANGLEMENT = REPO_ROOT / "Entanglement simulations"
COMPILATION = REPO_ROOT / "Compilation"
HEAT_MODELING = REPO_ROOT / "Heat_Modeling"
STRAIN_DRIVING = REPO_ROOT / "SiV Strain Driving"

# CSV files (exist in both Temperature and T2 and Entanglement folders)
N_VALUES = [2, 4, 8, 16]


def csv_path(N: int, folder: str = "temperature") -> Path:
    """Return path to N=X_results_parallel.csv."""
    base = TEMPERATURE_T2 if folder == "temperature" else ENTANGLEMENT
    return base / f"N={N}_results_parallel.csv"


def fid_csv_path() -> Path:
    """Return path to Fid_N=2.csv."""
    return TEMPERATURE_T2 / "Fid_N=2.csv"
