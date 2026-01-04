# src/perception/energy_bridge.py
"""
Day 08: Bridge between perception and frozen energy logic (Phase 1).

Goal:
Compute energy_ok using Phase 1 logic in src/energy (read-only).
If Phase 1 function is not found, fallback to an explicit threshold rule.
"""

from __future__ import annotations
import pandas as pd


def compute_energy_ok(timestamps: pd.Series, energy_available_wh: pd.Series) -> pd.Series:
    """
    Returns energy_ok as 1/0.

    IMPORTANT:
    - Preferred: call frozen Phase 1 energy logic from src/energy.
    - Fallback: simple threshold rule (kept explicit).
    """

    # --- Try common Phase 1 entrypoints (read-only imports) ---
    # Adjust these imports once you confirm the exact function name in src/energy.
    try:
        # Example A (if you have such a function)
        from src.energy.day04_energy_budget_scheduler import compute_energy_ok as phase1_energy_ok  # type: ignore
        return phase1_energy_ok(timestamps, energy_available_wh).astype(int)
    except Exception:
        pass

    try:
        # Example B (alternative naming)
        from src.energy.energy_windows import compute_energy_ok as phase1_energy_ok  # type: ignore
        return phase1_energy_ok(timestamps, energy_available_wh).astype(int)
    except Exception:
        pass

    # --- Fallback (explicit + deterministic) ---
    return (energy_available_wh >= 120).astype(int)
