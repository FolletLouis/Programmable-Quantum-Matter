"""
Data loading for the paper explorer.
Loads precomputed CSV files from the repo (fast mode).
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from .paths import TEMPERATURE_T2, ENTANGLEMENT, N_VALUES, csv_path


def load_t2_csv(N: int, folder: str = "temperature") -> Optional[pd.DataFrame]:
    """
    Load N=X_results_parallel.csv.
    Returns None if file not found (e.g., Filter_Function_II not yet run).
    """
    path = csv_path(N, folder)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_all_t2_csvs(folder: str = "temperature") -> dict[int, pd.DataFrame]:
    """Load all N=2,4,8,16 CSV files. Skips missing files."""
    result = {}
    for N in N_VALUES:
        df = load_t2_csv(N, folder)
        if df is not None:
            result[N] = df
    return result


def csvs_available() -> bool:
    """Check if at least one CSV exists (for fast mode)."""
    for N in N_VALUES:
        if csv_path(N).exists():
            return True
    return False
