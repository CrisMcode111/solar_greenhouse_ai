from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
import numpy as np


# --- Actions (minimal set) ---
# Ventilation: ON/OFF via intensity > 0
# Irrigation: ON/OFF via intensity > 0

@dataclass(frozen=True)
class RulesConfig:
    # Comfort thresholds
    temp_hi_c: float = 30.0
    temp_lo_c: float = 12.0
    rh_hi_pct: float = 85.0

    # Irrigation thresholds (if soil moisture exists)
    soil_moisture_lo: float = 0.42  # normalized 0..1, tune later

    # Risk flags handling
    risk_ventilate_flags: Tuple[str, ...] = (
    "risk_high_humidity",
    "risk_heat_stress",
    "rh_out_of_bounds",
    "mold_risk",
    "fungal_risk",
)

    risk_irrigate_flags: Tuple[str, ...] = ("water_stress", "drought_risk")

    # Energy gating
    allow_actuation_when_energy_not_ok: bool = False


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _as_bool_energy_ok(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "ok")
    try:
        return bool(int(val))
    except Exception:
        return bool(val)


def decide(
    state: Dict[str, Any],
    constraints: Dict[str, Any],
    risk_flags: List[str] | None,
    cfg: RulesConfig,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Returns:
      action: {vent_on: bool, vent_intensity: float, irrigate: bool, irrigate_intensity: float}
      why: human-readable rationale
      debug: small dict with signals used (for traceability)
    """
    energy_ok = _as_bool_energy_ok(constraints.get("energy_ok"))
    if (not energy_ok) and (not cfg.allow_actuation_when_energy_not_ok):
        action = {
            "vent_on": False,
            "vent_intensity": 0.0,
            "irrigate": False,
            "irrigate_intensity": 0.0,
        }
        why = "energy_ok=False → no actuation (conserve)"
        debug = {"energy_ok": energy_ok}
        return action, why, debug

    t = _to_float(state.get("inside_temp_c"))
    rh = _to_float(state.get("inside_rh_pct"))
    soil = _to_float(state.get("soil_moisture"))  # may be NaN if not present

    rf = risk_flags or []
    rf_set = set(rf)

    # --- Vent decision ---
    vent_on = False
    vent_intensity = 0.0
    vent_reasons: List[str] = []

    # risk-driven
    if any(flag in rf_set for flag in cfg.risk_ventilate_flags):
        vent_on = True
        vent_intensity = max(vent_intensity, 0.7)
        vent_reasons.append("risk_flag→ventilate")

    # threshold-driven
    if np.isfinite(t) and t > cfg.temp_hi_c:
        vent_on = True
        vent_intensity = max(vent_intensity, min(1.0, (t - cfg.temp_hi_c) / 10.0))
        vent_reasons.append(f"temp>{cfg.temp_hi_c}")

    if np.isfinite(rh) and rh > cfg.rh_hi_pct:
        vent_on = True
        vent_intensity = max(vent_intensity, min(1.0, (rh - cfg.rh_hi_pct) / 15.0))
        vent_reasons.append(f"rh>{cfg.rh_hi_pct}")

    # --- Irrigation decision ---
    irrigate = False
    irrigate_intensity = 0.0
    irrigate_reasons: List[str] = []

    # risk-driven
    if any(flag in rf_set for flag in cfg.risk_irrigate_flags):
        irrigate = True
        irrigate_intensity = max(irrigate_intensity, 0.6)
        irrigate_reasons.append("risk_flag→irrigate")

    # threshold-driven (only if soil exists)
    if np.isfinite(soil) and soil < cfg.soil_moisture_lo:
        irrigate = True
        irrigate_intensity = max(irrigate_intensity, min(1.0, (cfg.soil_moisture_lo - soil) / cfg.soil_moisture_lo))
        irrigate_reasons.append(f"soil<{cfg.soil_moisture_lo}")

    # Build why
    reasons = []
    if vent_on:
        reasons.append("VENT_ON(" + ",".join(vent_reasons) + f",int={vent_intensity:.2f})")
    else:
        reasons.append("VENT_OFF")

    if irrigate:
        reasons.append("IRRIGATE(" + ",".join(irrigate_reasons) + f",int={irrigate_intensity:.2f})")
    else:
        reasons.append("NO_IRRIGATE")

    why = " | ".join(reasons)
    debug = {
        "energy_ok": energy_ok,
        "temp_c": t,
        "rh_pct": rh,
        "soil_moisture": soil,
        "risk_flags": rf,
    }

    action = {
        "vent_on": bool(vent_on),
        "vent_intensity": float(vent_intensity),
        "irrigate": bool(irrigate),
        "irrigate_intensity": float(irrigate_intensity),
    }
    return action, why, debug


def config_to_json(cfg: RulesConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    # tuples to lists for JSON
    d["risk_ventilate_flags"] = list(cfg.risk_ventilate_flags)
    d["risk_irrigate_flags"] = list(cfg.risk_irrigate_flags)
    return d
