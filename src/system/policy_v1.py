from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class PolicyConfigV1:
    # Vent intensities
    vent_intensity_high_temp: float = 0.70
    vent_intensity_high_temp_hot_outside: float = 1.00
    vent_intensity_high_humid: float = 0.25
    vent_intensity_micro_cold: float = 0.15

    # Irrigation intensities
    irrigate_intensity_dry: float = 0.15
    irrigate_intensity_dry_hot: float = 0.10  # reduce due to evaporation risk

    # Hard gating
    allow_actuation_when_energy_not_ok: bool = False


def _has(token: str, tokens: list[str]) -> bool:
    return token in tokens


def policy_decide_from_state(
    state_code: str,
    cfg: PolicyConfigV1 | None = None
) -> Tuple[Dict[str, Any], str]:
    """
    Compute a policy decision from discrete state_code.
    Returns (action_dict, why_string).
    """
    cfg = cfg or PolicyConfigV1()
    tokens = state_code.split("|")

    action: Dict[str, Any] = {
        "vent_on": False,
        "vent_intensity": 0.0,
        "irrigate": False,
        "irrigate_intensity": 0.0,
    }
    why_parts: list[str] = []

    # --- Energy gate ---
    energy_ok = _has("E_OK", tokens)
    if (not energy_ok) and (not cfg.allow_actuation_when_energy_not_ok):
        return action, "policy: E_NOT_OK → conserve (no actuation)"

    # --- Outside safety for ventilation ---
    if _has("O_FREEZING", tokens):
        why_parts.append("policy: O_FREEZING → vent disabled (avoid cold shock)")
    elif _has("O_COLD", tokens):
        if _has("T_HIGH", tokens):
            action["vent_on"] = True
            action["vent_intensity"] = cfg.vent_intensity_micro_cold
            why_parts.append(f"policy: O_COLD & T_HIGH → micro-vent {cfg.vent_intensity_micro_cold:.2f}")
        else:
            why_parts.append("policy: O_COLD → no vent unless T_HIGH")
    else:
        # --- Normal vent logic ---
        if _has("T_HIGH", tokens):
            action["vent_on"] = True
            if _has("O_HOT", tokens):
                action["vent_intensity"] = cfg.vent_intensity_high_temp_hot_outside
                why_parts.append(f"policy: T_HIGH & O_HOT → vent {cfg.vent_intensity_high_temp_hot_outside:.2f}")
            else:
                action["vent_intensity"] = cfg.vent_intensity_high_temp
                why_parts.append(f"policy: T_HIGH → vent {cfg.vent_intensity_high_temp:.2f}")
        elif _has("H_HIGH", tokens) and (not _has("T_LOW", tokens)):
            action["vent_on"] = True
            action["vent_intensity"] = cfg.vent_intensity_high_humid
            why_parts.append(f"policy: H_HIGH & not T_LOW → vent {cfg.vent_intensity_high_humid:.2f}")
        else:
            why_parts.append("policy: no vent trigger")

    # --- Irrigation logic ---
    if _has("S_DRY", tokens):
        action["irrigate"] = True
        if _has("T_HIGH", tokens) and _has("O_HOT", tokens):
            action["irrigate_intensity"] = cfg.irrigate_intensity_dry_hot
            why_parts.append(f"policy: S_DRY but T_HIGH & O_HOT → irrigate {cfg.irrigate_intensity_dry_hot:.2f}")
        else:
            action["irrigate_intensity"] = cfg.irrigate_intensity_dry
            why_parts.append(f"policy: S_DRY → irrigate {cfg.irrigate_intensity_dry:.2f}")
    else:
        why_parts.append("policy: no irrigate trigger")

    return action, " | ".join(why_parts)
