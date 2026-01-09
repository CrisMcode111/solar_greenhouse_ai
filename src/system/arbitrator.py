from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.system.stress import compute_stress_from_state


@dataclass(frozen=True)
class ArbitrationConfig:
    # Hard gates
    forbid_vent_when_freezing: bool = True

    # Override thresholds
    override_to_policy_when_total_stress_ge: float = 0.7

    # If rules do nothing but stress is high, allow policy to act
    allow_policy_to_rescue_when_rules_idle: bool = True


def _is_acting(action: Dict[str, Any]) -> bool:
    return bool(action.get("vent_on") or action.get("irrigate"))


def arbitrate(
    state_code: str,
    rules_action: Dict[str, Any],
    rules_why: str,
    policy_action: Dict[str, Any],
    policy_why: str,
    cfg: ArbitrationConfig | None = None,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Returns (final_action, final_why, meta)
    meta includes decision_source and stress.
    """
    cfg = cfg or ArbitrationConfig()
    tokens = state_code.split("|")
    stress = compute_stress_from_state(state_code)

    # Start from rules by default (keeps continuity with Day12 baseline)
    final_action = dict(rules_action)
    meta: Dict[str, Any] = {
        "decision_source": "rules",
        "override": False,
        "stress": stress,
    }

    # --- Hard energy gate: if NOT_OK then no actuation regardless ---
    if "E_NOT_OK" in tokens:
        final_action = {
            "vent_on": False,
            "vent_intensity": 0.0,
            "irrigate": False,
            "irrigate_intensity": 0.0,
        }
        meta["decision_source"] = "gated_energy"
        meta["override"] = True
        return final_action, "gate: E_NOT_OK → no actuation", meta

    # --- Hard freezing gate: vent forbidden ---
    if cfg.forbid_vent_when_freezing and ("O_FREEZING" in tokens):
        if final_action.get("vent_on"):
            final_action["vent_on"] = False
            final_action["vent_intensity"] = 0.0
            meta["decision_source"] = "gated_freezing"
            meta["override"] = True
            return final_action, "gate: O_FREEZING → vent disabled", meta

    # --- Soft arbitration: if stress high, prefer policy ---
    total_stress = float(stress.get("total_stress", 0.0))

    rules_acting = _is_acting(rules_action)
    policy_acting = _is_acting(policy_action)

    if total_stress >= cfg.override_to_policy_when_total_stress_ge:
        # If policy acts, switch to policy (more contextual)
        if policy_acting:
            final_action = dict(policy_action)
            meta["decision_source"] = "policy_override_high_stress"
            meta["override"] = True
            return final_action, f"override: high stress ({total_stress:.2f}) → policy | {policy_why}", meta

        # If policy is idle but rules act, keep rules
        if rules_acting:
            return final_action, f"keep rules: stress high but policy idle | rules: {rules_why}", meta

        # Both idle: remain idle
        return final_action, f"both idle under high stress | rules: {rules_why} | policy: {policy_why}", meta

    # --- Rescue mode: rules idle, policy can act (even if stress moderate) ---
    if cfg.allow_policy_to_rescue_when_rules_idle and (not rules_acting) and policy_acting:
        final_action = dict(policy_action)
        meta["decision_source"] = "policy_rescue"
        meta["override"] = True
        return final_action, f"rescue: rules idle → policy | {policy_why}", meta

    # Default: keep rules
    return final_action, f"keep rules | rules: {rules_why}", meta
