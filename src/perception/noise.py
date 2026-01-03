# src/perception/noise.py
"""
Phase 2 – Perception
Day 07: Noise & sensor corruptions

Applies realistic imperfections to a clean sensor dataframe:
- random missing values
- missing blocks (consecutive gaps)
- spikes / outliers
- slow drift (sensor bias accumulating over time)

This module MUST NOT generate base signals.
It only corrupts existing sensor dataframes.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NoiseConfig:
    # Missingness
    missing_rate: float = 0.02              # random missing points per column
    missing_block_rate: float = 0.002       # probability of starting a missing block at a timestep
    missing_block_min_len: int = 2
    missing_block_max_len: int = 8

    # Spikes / outliers
    spike_rate: float = 0.005               # probability of spike at a timestep
    spike_std_map: dict = None              # per-column spike std (fallback default if missing)

    # Drift
    drift_per_day_map: dict = None          # per-column drift per day (additive bias)

    # General
    seed: int = 42


def _default_spike_std_map() -> dict:
    return {
        "inside_temp_c": 2.5,
        "outside_temp_c": 2.0,
        "inside_rh_pct": 10.0,
        "outside_rh_pct": 8.0,
        "soil_moisture_pct": 6.0,
        "light_lux": 25000.0,
        "energy_available_wh": 250.0,
    }


def _default_drift_map() -> dict:
    return {
        # example: a thermometer slowly drifts +0.02°C per day
        "inside_temp_c": 0.02,
        # example: humidity sensor bias
        "inside_rh_pct": 0.05,
    }


def apply_noise(df: pd.DataFrame, cfg: NoiseConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    out = df.copy()

    n = len(out)
    if n == 0:
        return out

    spike_std_map = cfg.spike_std_map or _default_spike_std_map()
    drift_map = cfg.drift_per_day_map or _default_drift_map()

    # choose numeric columns that exist
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]

    # --- 1) Drift (additive bias accumulating over time)
    hours = np.arange(n, dtype=float)
    for col, drift_per_day in drift_map.items():
        if col in out.columns:
            drift = (hours / 24.0) * float(drift_per_day)
            out[col] = out[col].astype(float) + drift

    # --- 2) Random missing values (pointwise)
    for col in numeric_cols:
        mask = rng.random(n) < cfg.missing_rate
        out.loc[mask, col] = np.nan

    # --- 3) Missing blocks (consecutive gaps)
    for col in numeric_cols:
        starts = np.where(rng.random(n) < cfg.missing_block_rate)[0]
        for s in starts:
            block_len = int(rng.integers(cfg.missing_block_min_len, cfg.missing_block_max_len + 1))
            e = min(n, s + block_len)
            out.loc[out.index[s:e], col] = np.nan

    # --- 4) Spikes / outliers
    for col, spike_std in spike_std_map.items():
        if col not in out.columns:
            continue
        mask = rng.random(n) < cfg.spike_rate
        if mask.any():
            spikes = rng.normal(0.0, float(spike_std), size=int(mask.sum()))
            out.loc[mask, col] = out.loc[mask, col].astype(float) + spikes

    # --- 5) Clamp typical physical ranges (only if columns exist)
    def _clip(col: str, lo: float, hi: float):
        if col in out.columns:
            out[col] = out[col].astype(float).clip(lo, hi)

    _clip("inside_temp_c", -10, 60)
    _clip("outside_temp_c", -25, 50)
    _clip("inside_rh_pct", 0, 100)
    _clip("outside_rh_pct", 0, 100)
    _clip("soil_moisture_pct", 0, 100)
    if "light_lux" in out.columns:
        out["light_lux"] = out["light_lux"].astype(float).clip(0, None)
    if "energy_available_wh" in out.columns:
        out["energy_available_wh"] = out["energy_available_wh"].astype(float).clip(0, None)

    return out
