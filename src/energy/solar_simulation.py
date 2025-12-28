"""
Solar energy simulation utilities.

This module provides a lightweight, explainable approximation of daily
solar energy availability. It is intentionally dataset-free and suitable
for frugal AI experimentation on low-resource systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SolarSimConfig:
    """
    Configuration for a one-day solar energy simulation.
    """
    sunrise: float = 7.0          # hour in [0, 24]
    sunset: float = 19.0          # hour in [0, 24]
    cloud_strength: float = 0.15  # 0 = clear sky
    smooth_k: int = 5             # moving average window size
    step_minutes: int = 5         # temporal resolution
    seed: int = 42                # reproducibility


def make_time_grid(
    start: Optional[datetime] = None,
    step_minutes: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Build a 24h time grid at the desired resolution.

    Returns a dict with:
    - times: np.ndarray of datetime objects (dtype=object)
    - hour_float: np.ndarray of floats (e.g., 13.5 for 13:30)
    """
    if start is None:
        start = datetime(2025, 1, 1, 0, 0, 0)

    minutes = np.arange(0, 24 * 60, step_minutes)
    times = np.array([start + timedelta(minutes=int(m)) for m in minutes], dtype=object)
    hour_float = np.array([t.hour + t.minute / 60 for t in times], dtype=float)

    return {"times": times, "hour_float": hour_float}


def simulate_solar_energy(
    hour_float: np.ndarray,
    sunrise: float,
    sunset: float,
) -> np.ndarray:
    """
    Compute an idealized half-sine solar curve between sunrise and sunset.
    Output is normalized in [0, 1].
    """
    if not (0.0 <= sunrise <= 24.0 and 0.0 <= sunset <= 24.0):
        raise ValueError("sunrise and sunset must be within [0, 24].")
    if sunset <= sunrise:
        raise ValueError("sunset must be greater than sunrise.")

    daylen = sunset - sunrise
    E = np.zeros_like(hour_float, dtype=float)

    mask_day = (hour_float >= sunrise) & (hour_float <= sunset)
    phase = (hour_float[mask_day] - sunrise) / daylen * np.pi
    E[mask_day] = np.sin(phase)

    return np.clip(E, 0.0, 1.0)


def add_cloud_variability(
    E_ideal: np.ndarray,
    cloud_strength: float,
    smooth_k: int,
    seed: int,
) -> np.ndarray:
    """
    Add smoothed stochastic variability ("clouds") to an ideal solar signal.
    """
    if cloud_strength < 0:
        raise ValueError("cloud_strength must be >= 0.")
    if smooth_k < 1:
        raise ValueError("smooth_k must be >= 1.")

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=cloud_strength, size=E_ideal.shape)

    # Smooth noise with a moving average
    kernel = np.ones(smooth_k) / smooth_k
    noise_smooth = np.convolve(noise, kernel, mode="same")

    E = np.clip(E_ideal + noise_smooth, 0.0, 1.0)
    return E


def simulate_solar_day(
    config: SolarSimConfig,
    start: Optional[datetime] = None,
) -> Dict[str, np.ndarray]:
    """
    End-to-end helper:
    - builds the time grid
    - computes ideal energy
    - adds cloud variability
    Returns a dict containing:
    - times
    - hour_float
    - E_solar_ideal
    - E_solar
    """
    grid = make_time_grid(start=start, step_minutes=config.step_minutes)
    hour_float = grid["hour_float"]

    E_ideal = simulate_solar_energy(
        hour_float=hour_float,
        sunrise=config.sunrise,
        sunset=config.sunset,
    )

    E = add_cloud_variability(
        E_ideal=E_ideal,
        cloud_strength=config.cloud_strength,
        smooth_k=config.smooth_k,
        seed=config.seed,
    )

    return {
        "times": grid["times"],
        "hour_float": hour_float,
        "E_solar_ideal": E_ideal,
        "E_solar": E,
    }
