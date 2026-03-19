"""
Configuration for paper explorer: fast vs full mode.
"""
from .data_loader import csvs_available

# FAST MODE: Use precomputed CSV, cached plots. No Filter_Function_II, no SAFE-GRAPE.
# FULL MODE: Would run Filter_Function_II (expensive, ~minutes for full sweep).
# For now we default to fast; full mode requires manual CSV generation.
FAST_MODE = True  # Use cached/precomputed data


def can_run_full_mode() -> bool:
    """True if CSV files exist (Filter_Function_II has been run)."""
    return csvs_available()
