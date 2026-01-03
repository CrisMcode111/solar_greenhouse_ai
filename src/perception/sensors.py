# src/perception/sensors.py
"""
Phase 2 – Perception
Day 06: Synthetic Sensors (generator time-series for greenhouse sensors)

Energy logic (Phase 1) is frozen in src/energy and will be imported (read-only).
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SensorGenConfig:
    start: str
    end: str
    freq: str = "1H"
    tz: str = "Europe/Paris"
    seed: int = 42


def generate_synthetic_sensors(cfg: SensorGenConfig) -> pd.DataFrame:
    """
    Generate a synthetic greenhouse sensor dataset (base/clean version).
    Noise/corruptions will be applied in src/perception/noise.py.
    """
    raise NotImplementedError("Day 06: implement generator logic next.")
