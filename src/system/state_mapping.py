from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math


# ----------------------------
# Discrete state vocabulary
# ----------------------------
TEMP_LOW = "LOW"
TEMP_OK = "OK"
TEMP_HIGH = "HIGH"

HUMID_LOW = "LOW"
HUMID_OK = "OK"
HUMID_HIGH = "HIGH"

SOIL_DRY = "DRY"
SOIL_OK = "OK"
SOIL_WET = "WET"

ENERGY_OK = "OK"
ENERGY_NOT_OK = "NOT_OK"

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class StateMappingConfig:
    """
    Aligned with Day12 rules:
      - temp_hi_c = 30.0
      - rh_hi_pct = 85.0
      - soil_moisture_lo = 0.42 (after normalization 0..1)
    Also includes reasonable 'low' and 'wet' thresholds to create 3-bin states.
    """

    # Temperature thresholds (°C)
    temp_lo_c: float = 12.0
    temp_hi_c: float = 30.0

    # Relative humidity thresholds (%)
    rh_lo_pct: float = 35.0
    rh_hi_pct: float = 85.0

    # Soil moisture thresholds (normalized 0..1)
    soil_lo: float = 0.42
    soil_hi: float = 0.70

    # If soil comes in 0..100, normalize to 0..1
    normalize_soil_over: float = 1.5

    # Formatting of state codes
    sep: str = "|"
    prefix_temp: str = "T"
    prefix_soil: str = "S"
    prefix_humid: str = "H"
    prefix_energy: str = "E"


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def normalize_soil_moisture(x: Any, cfg: StateMappingConfig) -> float:
    """
    Normalize soil moisture to 0..1 when needed.
    - if value is > cfg.normalize_soil_over, assume 0..100 and divide by 100.
    Returns float('nan') if missing/unparseable.
    """
    v = _to_float(x)
    if not _is_finite(v):
        return float("nan")
    if v > cfg.normalize_soil_over:
        return v / 100.0
    return v


def map_temp(temp_c: Any, cfg: StateMappingConfig) -> Tuple[str, List[str]]:
    v = _to_float(temp_c)
    reasons: List[str] = []
    if not _is_finite(v):
        return UNKNOWN, ["temp missing/unparseable"]

    if v < cfg.temp_lo_c:
        reasons.append(f"temp {v:.2f} < {cfg.temp_lo_c:.2f} (low)")
        return TEMP_LOW, reasons
    if v > cfg.temp_hi_c:
        reasons.append(f"temp {v:.2f} > {cfg.temp_hi_c:.2f} (high)")
        return TEMP_HIGH, reasons

    reasons.append(f"temp {v:.2f} within [{cfg.temp_lo_c:.2f}, {cfg.temp_hi_c:.2f}] (ok)")
    return TEMP_OK, reasons


def map_humid(rh_pct: Any, cfg: StateMappingConfig) -> Tuple[str, List[str]]:
    v = _to_float(rh_pct)
    reasons: List[str] = []
    if not _is_finite(v):
        return UNKNOWN, ["humidity missing/unparseable"]

    if v < cfg.rh_lo_pct:
        reasons.append(f"rh {v:.2f}% < {cfg.rh_lo_pct:.2f}% (low)")
        return HUMID_LOW, reasons
    if v > cfg.rh_hi_pct:
        reasons.append(f"rh {v:.2f}% > {cfg.rh_hi_pct:.2f}% (high)")
        return HUMID_HIGH, reasons

    reasons.append(f"rh {v:.2f}% within [{cfg.rh_lo_pct:.2f}%, {cfg.rh_hi_pct:.2f}%] (ok)")
    return HUMID_OK, reasons


def map_soil(soil_moisture: Any, cfg: StateMappingConfig) -> Tuple[str, List[str], float]:
    """
    Returns (soil_state, reasons, soil_norm).
    soil_norm is normalized 0..1 (or NaN).
    """
    soil_norm = normalize_soil_moisture(soil_moisture, cfg)
    reasons: List[str] = []

    if not _is_finite(soil_norm):
        return UNKNOWN, ["soil moisture missing/unparseable"], soil_norm

    if soil_norm < cfg.soil_lo:
        reasons.append(f"soil {soil_norm:.3f} < {cfg.soil_lo:.3f} (dry)")
        return SOIL_DRY, reasons, soil_norm
    if soil_norm > cfg.soil_hi:
        reasons.append(f"soil {soil_norm:.3f} > {cfg.soil_hi:.3f} (wet)")
        return SOIL_WET, reasons, soil_norm

    reasons.append(f"soil {soil_norm:.3f} within [{cfg.soil_lo:.3f}, {cfg.soil_hi:.3f}] (ok)")
    return SOIL_OK, reasons, soil_norm


def map_energy(energy_ok: Any) -> Tuple[str, List[str]]:
    """
    Accepts bool/int/str.
    """
    reasons: List[str] = []
    if energy_ok is None:
        return UNKNOWN, ["energy_ok missing"]

    if isinstance(energy_ok, bool):
        state = ENERGY_OK if energy_ok else ENERGY_NOT_OK
        reasons.append(f"energy_ok={energy_ok}")
        return state, reasons

    if isinstance(energy_ok, str):
        s = energy_ok.strip().lower()
        ok = s in ("1", "true", "yes", "ok")
        state = ENERGY_OK if ok else ENERGY_NOT_OK
        reasons.append(f"energy_ok='{energy_ok}' → {ok}")
        return state, reasons

    try:
        ok = bool(int(energy_ok))
        state = ENERGY_OK if ok else ENERGY_NOT_OK
        reasons.append(f"energy_ok={energy_ok} → {ok}")
        return state, reasons
    except Exception:
        ok = bool(energy_ok)
        state = ENERGY_OK if ok else ENERGY_NOT_OK
        reasons.append(f"energy_ok={energy_ok} (bool cast) → {ok}")
        return state, reasons


def make_state_code(
    temp_state: str,
    soil_state: str,
    humid_state: str,
    energy_state: str,
    cfg: StateMappingConfig,
) -> str:
    return cfg.sep.join(
        [
            f"{cfg.prefix_temp}_{temp_state}",
            f"{cfg.prefix_soil}_{soil_state}",
            f"{cfg.prefix_humid}_{humid_state}",
            f"{cfg.prefix_energy}_{energy_state}",
        ]
    )


def map_state(
    raw: Dict[str, Any],
    cfg: StateMappingConfig | None = None,
) -> Dict[str, Any]:
    """
    Map numeric/raw signals to discrete states + explanations.

    Expected keys in `raw` (flexible; missing is allowed):
      - inside_temp_c
      - inside_rh_pct
      - soil_moisture
      - energy_ok

    Returns dict with:
      - temp_state, humid_state, soil_state, energy_state
      - soil_moisture_norm (float or NaN)
      - state_code
      - explanation (string)
      - explanation_parts (list of strings)
    """
    cfg = cfg or StateMappingConfig()

    temp_state, temp_reasons = map_temp(raw.get("inside_temp_c"), cfg)
    humid_state, humid_reasons = map_humid(raw.get("inside_rh_pct"), cfg)
    soil_state, soil_reasons, soil_norm = map_soil(raw.get("soil_moisture"), cfg)
    energy_state, energy_reasons = map_energy(raw.get("energy_ok"))

    state_code = make_state_code(temp_state, soil_state, humid_state, energy_state, cfg)

    parts: List[str] = []
    parts.extend(temp_reasons)
    parts.extend(humid_reasons)
    parts.extend(soil_reasons)
    parts.extend(energy_reasons)

    explanation = "; ".join(parts)

    return {
        "temp_state": temp_state,
        "humid_state": humid_state,
        "soil_state": soil_state,
        "energy_state": energy_state,
        "soil_moisture_norm": soil_norm,
        "state_code": state_code,
        "explanation": explanation,
        "explanation_parts": parts,
    }
