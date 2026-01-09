from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.system.utils_io import ensure_dir, write_json, utc_now_iso, env_fingerprint


@dataclass(frozen=True)
class PolicyConfig:
    """
    A simple baseline policy operating on discrete state codes.
    This is NOT ML. It's a heuristic policy used for comparison.
    """

    # If energy is NOT OK, policy does nothing (same safety constraint as rules)
    allow_actuation_when_energy_not_ok: bool = False

    # Policy choices: be more "risk-averse" to heat, more "water-saving" on irrigation
    irrigate_only_if_soil_dry: bool = True

    # Intensity baselines
    vent_intensity_high_temp: float = 0.8
    vent_intensity_ok_temp_high_humid: float = 0.4
    irrigate_intensity_dry: float = 0.10


def parse_state_code(state_code: str) -> Dict[str, str]:
    """
    Expects format like: T_HIGH|S_DRY|H_OK|E_OK
    Returns dict: {"T": "HIGH", "S": "DRY", "H": "OK", "E": "OK"}
    """
    out: Dict[str, str] = {}
    if not isinstance(state_code, str):
        return out
    parts = [p.strip() for p in state_code.split("|")]
    for p in parts:
        if "_" not in p:
            continue
        k, v = p.split("_", 1)
        out[k.strip()] = v.strip()
    return out


def policy_decide(state_code: str, cfg: PolicyConfig) -> Tuple[Dict[str, Any], str]:
    """
    Returns (policy_action, why).
    policy_action fields match rules fields for easy comparison.
    """
    s = parse_state_code(state_code)

    t = s.get("T", "UNKNOWN")
    soil = s.get("S", "UNKNOWN")
    h = s.get("H", "UNKNOWN")
    e = s.get("E", "UNKNOWN")
    o = s.get("O", "UNKNOWN")

    energy_ok = (e == "OK")

    # default: do nothing
    action = {
        "vent_on": False,
        "vent_intensity": 0.0,
        "irrigate": False,
        "irrigate_intensity": 0.0,
    }

    if (not energy_ok) and (not cfg.allow_actuation_when_energy_not_ok):
        return action, "policy: energy NOT_OK → conserve"

    why_parts = []

    # Vent policy:
    # - If temp HIGH: ventilate strongly
    # - Else if humid HIGH (and temp not LOW): ventilate moderately
    # Vent policy (season-aware):
    # - If outside is FREEZING: do not ventilate (safety)
    # - If outside is COLD: micro-vent only if inside is HIGH
    # - Else (MILD/HOT/UNKNOWN): normal vent logic
    if o == "FREEZING":
        why_parts.append("policy: O=FREEZING → vent disabled (avoid cold shock)")
    elif o == "COLD":
        if t == "HIGH":
            action["vent_on"] = True
            action["vent_intensity"] = 0.15
            why_parts.append("policy: O=COLD & T=HIGH → micro-vent 0.15")
        else:
            why_parts.append("policy: O=COLD → no vent unless T=HIGH")
    else:
        if t == "HIGH":
            action["vent_on"] = True
            action["vent_intensity"] = cfg.vent_intensity_high_temp
            why_parts.append(f"policy: T={t} → vent {cfg.vent_intensity_high_temp:.2f}")
        elif h == "HIGH" and t != "LOW":
            action["vent_on"] = True
            action["vent_intensity"] = cfg.vent_intensity_ok_temp_high_humid
            why_parts.append(f"policy: H={h} & T={t} → vent {cfg.vent_intensity_ok_temp_high_humid:.2f}")
        else:
            why_parts.append("policy: no vent rule triggered")

        return action, " | ".join(why_parts)

    # Irrigation policy:


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/day13/state_action_timeline.csv", type=str)
    ap.add_argument("--out_dir", default="artifacts/day14", type=str)
    ap.add_argument("--max_rows", default=0, type=int, help="0 means all rows")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    df = pd.read_csv(Path(args.input))
    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows)).copy()

    # compute policy actions per row
    cfg = PolicyConfig()

    policy_rows = []
    for _, r in df.iterrows():
        state_code = r.get("state_code", "")
        act, why = policy_decide(state_code, cfg)
        policy_rows.append(
            {
                "step_idx": r.get("step_idx"),
                "ts": r.get("ts"),
                "state_code": state_code,
                "policy_vent_on": act["vent_on"],
                "policy_vent_intensity": act["vent_intensity"],
                "policy_irrigate": act["irrigate"],
                "policy_irrigate_intensity": act["irrigate_intensity"],
                "policy_why": why,
            }
        )

    dfp = pd.DataFrame(policy_rows)

    # merge policy actions back
    merged = df.merge(dfp, on=["step_idx", "ts", "state_code"], how="left", validate="one_to_one")

    # agreement metrics (boolean decisions)
    def to_bool_series(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "ok"])

    rules_vent = to_bool_series(merged["vent_on"])
    rules_irr = to_bool_series(merged["irrigate"])
    pol_vent = to_bool_series(merged["policy_vent_on"])
    pol_irr = to_bool_series(merged["policy_irrigate"])

    merged["agree_vent"] = (rules_vent == pol_vent)
    merged["agree_irrigate"] = (rules_irr == pol_irr)
    merged["agree_both"] = (merged["agree_vent"] & merged["agree_irrigate"])

    # disagreement table by state
    by_state = (
        merged.groupby("state_code")
        .agg(
            count=("step_idx", "count"),
            agree_both_rate=("agree_both", "mean"),
            disagree_vent_rate=("agree_vent", lambda x: 1.0 - float(x.mean())),
            disagree_irrigate_rate=("agree_irrigate", lambda x: 1.0 - float(x.mean())),
        )
        .sort_values(["disagree_vent_rate", "disagree_irrigate_rate", "count"], ascending=False)
        .reset_index()
    )

    # output files
    policy_csv = out_dir / "policy_actions.csv"
    comparison_csv = out_dir / "rules_vs_policy_timeline.csv"
    disagree_csv = out_dir / "disagreement_by_state.csv"
    summary_json = out_dir / "comparison_summary.json"

    dfp.to_csv(policy_csv, index=False)

    # keep the most useful columns for timeline comparison
    keep_cols = [
        "step_idx",
        "ts",
        "state_code",
        "vent_on",
        "vent_intensity",
        "irrigate",
        "irrigate_intensity",
        "why",
        "policy_vent_on",
        "policy_vent_intensity",
        "policy_irrigate",
        "policy_irrigate_intensity",
        "policy_why",
        "agree_vent",
        "agree_irrigate",
        "agree_both",
    ]
    merged[keep_cols].to_csv(comparison_csv, index=False)

    by_state.to_csv(disagree_csv, index=False)

    write_json(
        summary_json,
        {
            "day": "day14",
            "created_utc": utc_now_iso(),
            "rows": int(len(merged)),
            "agreement_vent_rate": float(merged["agree_vent"].mean()),
            "agreement_irrigate_rate": float(merged["agree_irrigate"].mean()),
            "agreement_both_rate": float(merged["agree_both"].mean()),
            "top_disagreement_states": by_state.head(10).to_dict(orient="records"),
            "env": env_fingerprint(),
        },
    )

    print(f"[Day14] Done. Artifacts at: {out_dir}")
    print(f"[Day14] Wrote: {policy_csv.name}, {comparison_csv.name}, {disagree_csv.name}, {summary_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
