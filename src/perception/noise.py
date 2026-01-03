# src/perception/noise.py
"""
Phase 2 – Perception
Noise & sensor corruption models.

Applies realistic imperfections:
- missing values
- spikes / outliers
- sensor drift

This module MUST NOT generate base signals.
It only corrupts existing sensor dataframes.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseConfig:
    missing_rate: float = 0.02
    spike_rate: float = 0.005
    drift_per_day: float = 0.0


def apply_noise(df: pd.DataFrame, cfg: NoiseConfig) -> pd.DataFrame:
    """
    Apply noise and corruption to a clean sensor dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Clean sensor data from sensors.generate_synthetic_sensors
    cfg : NoiseConfig
        Noise parameters

    Returns
    -------
    pd.DataFrame
        Corrupted sensor dataframe
    """
    raise NotImplementedError("Day 06: noise logic to be implemented.")
