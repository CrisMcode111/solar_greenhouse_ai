# src/perception/sensors.py
"""
Phase 2 – Perception
Day 06: Synthetic Sensors (clean/base generator)

Generates a synthetic hourly time-series for a greenhouse:
- outside/inside temperature
- outside/inside humidity
- light (lux)
- soil moisture + irrigation events
- vent state
- energy_available_wh + energy_ok (placeholder threshold)

Noise/corruptions are applied later in src/perception/noise.py
Energy logic (Phase 1) remains frozen in src/energy/ and can replace energy_ok later.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorGenConfig:
    start: str
    end: str
    freq: str = "1h"
    tz: str = "Europe/Paris"
    seed: int = 42


def _solar_shape(hour: float, day_length_h: float) -> float:
    """
    Smooth bell-like curve:
    0 at night, peak near noon, scaled by day length.
    """
    noon = 12.0
    half = max(day_length_h / 2.0, 1e-6)
    x = (hour - noon) / half
    return float(max(0.0, 1.0 - x * x))


def generate_synthetic_sensors(cfg: SensorGenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    ts = pd.date_range(
        start=cfg.start,
        end=cfg.end,
        freq=cfg.freq,
        tz=cfg.tz,
        inclusive="left",
    )
    if len(ts) < 2:
        raise ValueError("Time range too small. Increase end-start or use smaller freq.")

    df = pd.DataFrame({"timestamp": ts})
    df["day_of_year"] = df["timestamp"].dt.dayofyear.astype(int)
    df["hour"] = df["timestamp"].dt.hour.astype(int)

    doy = df["day_of_year"].to_numpy()
    hour = df["hour"].to_numpy().astype(float)

    # Day length proxy across year (winter ~ 8.5h, summer ~ 16h)
    day_length = 12 + 3.75 * np.sin(2 * np.pi * (doy - 80) / 365.0)

    # Solar / light
    solar = np.array([_solar_shape(h, dl) for h, dl in zip(hour, day_length)])
    clouds = rng.normal(1.0, 0.15, size=len(df)).clip(0.4, 1.2)
    light_lux = 80000 * solar * clouds
    df["light_lux"] = np.round(light_lux, 0)

    # Energy proxy (Wh)
    eff = rng.normal(1.0, 0.10, size=len(df)).clip(0.7, 1.3)
    energy_wh = (df["light_lux"].to_numpy() / 80000.0) * 600.0 * eff
    df["energy_available_wh"] = np.clip(energy_wh, 0, None)

    # Outside temperature (season + daily wave + noise)
    seasonal_mean = 14 + 8 * np.sin(2 * np.pi * (doy - 172) / 365.0)  # ~6..22
    daily_amp = 5 + 2 * solar
    outside_temp = (
        seasonal_mean
        + daily_amp * np.sin(2 * np.pi * (hour - 6) / 24.0)
        + rng.normal(0, 0.8, len(df))
    )
    df["outside_temp_c"] = np.round(outside_temp, 2)

    # Outside humidity (rough inverse relation with daily heating)
    outside_rh = 75 - 1.5 * (outside_temp - seasonal_mean) + rng.normal(0, 4.0, len(df))
    df["outside_rh_pct"] = np.round(np.clip(outside_rh, 25, 98), 1)

    # Inside temperature: inertia + solar gain - vent cooling
    vent = np.zeros(len(df), dtype=int)
    inside = np.zeros(len(df), dtype=float)
    inside[0] = outside_temp[0] + 2.0

    alpha = 0.12         # coupling to outside (inertia)
    solar_gain_k = 6.0   # heating at peak solar
    vent_cool_k = 1.8    # cooling when vent ON

    for t in range(1, len(df)):
        # simple vent rule
        vent[t] = 1 if inside[t - 1] > 28.0 else 0
        inside[t] = (
            inside[t - 1]
            + alpha * (outside_temp[t] - inside[t - 1])
            + solar_gain_k * solar[t]
            - vent_cool_k * vent[t]
            + rng.normal(0, 0.25)
        )

    df["vent_state"] = vent
    df["inside_temp_c"] = np.round(inside, 2)

    # Soil moisture + irrigation events
    soil = np.zeros(len(df), dtype=float)
    soil[0] = 42.0
    irrigation = np.zeros(len(df), dtype=int)

    for t in range(1, len(df)):
        # irrigate if too dry in the morning
        if soil[t - 1] < 28 and df["hour"].iat[t] in (6, 7, 8):
            irrigation[t] = 1

        evap = 0.25 + 0.15 * solar[t] + 0.03 * max(df["inside_temp_c"].iat[t] - 18, 0)
        soil[t] = soil[t - 1] - evap + (12.0 if irrigation[t] else 0.0) + rng.normal(0, 0.2)
        soil[t] = float(np.clip(soil[t], 12, 55))

    df["irrigation_event"] = irrigation
    df["soil_moisture_pct"] = np.round(soil, 2)

    # Inside RH: influenced by soil, temp gradient, and vent
    inside_rh = (
        df["outside_rh_pct"].to_numpy()
        + 0.35 * (df["soil_moisture_pct"].to_numpy() - 35)
        - 1.2 * (df["inside_temp_c"].to_numpy() - df["outside_temp_c"].to_numpy())
        - 6.0 * df["vent_state"].to_numpy()
        + rng.normal(0, 3.0, len(df))
    )
    df["inside_rh_pct"] = np.round(np.clip(inside_rh, 20, 99), 1)

    # Energy OK (placeholder; replace with frozen Phase 1 later)
    df["energy_ok"] = (df["energy_available_wh"] >= 120).astype(int)

    return df
