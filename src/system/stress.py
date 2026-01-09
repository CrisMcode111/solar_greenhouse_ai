from __future__ import annotations
from typing import Dict


def compute_stress_from_state(state_code: str) -> Dict[str, float | str]:
    """
    Compute simple plant stress indicators from discrete state_code.

    Expected tokens in state_code:
      T_LOW | T_OK | T_HIGH
      S_DRY | S_OK | S_WET
      H_LOW | H_OK | H_HIGH
      E_OK | E_NOT_OK
      O_FREEZING | O_COLD | O_MILD | O_HOT
    """

    tokens = state_code.split("|")

    heat_stress = 0.0
    water_stress = 0.0
    cold_vent_risk = 0.0
    notes = []

    # --- Heat stress ---
    if "T_HIGH" in tokens:
        heat_stress = 0.7
        notes.append("heat stress: inside temp HIGH")

        if "O_HOT" in tokens:
            heat_stress = 1.0
            notes.append("outside HOT → amplified heat stress")

    # --- Water stress ---
    if "S_DRY" in tokens:
        water_stress = 0.7
        notes.append("water stress: soil DRY")

        if "T_HIGH" in tokens:
            water_stress = 1.0
            notes.append("high temp + dry soil → amplified water stress")

    # --- Cold risk for ventilation ---
    if "O_FREEZING" in tokens:
        cold_vent_risk = 1.0
        notes.append("outside FREEZING → vent forbidden")
    elif "O_COLD" in tokens:
        cold_vent_risk = 0.5
        notes.append("outside COLD → vent risky")

    total_stress = max(heat_stress, water_stress)

    return {
        "heat_stress": heat_stress,
        "water_stress": water_stress,
        "cold_vent_risk": cold_vent_risk,
        "total_stress": total_stress,
        "stress_notes": "; ".join(notes),
    }
