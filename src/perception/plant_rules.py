# src/perception/plant_rules.py
"""
Phase 2 – Perception
Rule-based plant / greenhouse heuristics.

Goal:
- derive simple risk flags from sensor data
- keep rules transparent & easy to adjust later
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class PlantRuleConfig:
    heat_stress_temp_c: float = 32.0
    cold_stress_temp_c: float = 8.0
    dry_soil_pct: float = 20.0
    high_humidity_pct: float = 90.0


def compute_risk_flags(df: pd.DataFrame, cfg: PlantRuleConfig) -> pd.DataFrame:
    """
    Adds simple boolean/int flags based on sensor columns.

    Expected columns (if present):
    - inside_temp_c
    - soil_moisture_pct
    - inside_rh_pct
    """
    out = df.copy()

    if "inside_temp_c" in out.columns:
        out["risk_heat_stress"] = (out["inside_temp_c"] >= cfg.heat_stress_temp_c).astype(int)
        out["risk_cold_stress"] = (out["inside_temp_c"] <= cfg.cold_stress_temp_c).astype(int)

    if "soil_moisture_pct" in out.columns:
        out["risk_dry_soil"] = (out["soil_moisture_pct"] <= cfg.dry_soil_pct).astype(int)

    if "inside_rh_pct" in out.columns:
        out["risk_high_humidity"] = (out["inside_rh_pct"] >= cfg.high_humidity_pct).astype(int)

    return out
