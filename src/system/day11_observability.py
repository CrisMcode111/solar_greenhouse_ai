from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.system.utils_io import ensure_dir, read_dataset, write_json, utc_now_iso, env_fingerprint
from src.system.schema import check_schema
from src.system.logger import SystemLogger
from src.system.metrics import write_metrics_summary


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_state(row: pd.Series, colmap: Dict[str, str]) -> Dict[str, Any]:
    def get(name: str):
        c = colmap.get(name)
        return None if c is None else row.get(c)

    return {
        "timestamp": get("timestamp"),
        "inside_temp_c": get("inside_temp_c"),
        "inside_rh_pct": get("inside_rh_pct"),
        "light_lux": get("light_lux"),
        "energy_available_wh": get("energy_available_wh"),
    }


def decide_action(state: Dict[str, Any], energy_ok: bool) -> Tuple[Dict[str, Any], str]:
    """
    Rule-based placeholder (Day11): we log full rationale.
    Day12/13 will improve + compare policy.
    """
    t = state.get("inside_temp_c")
    rh = state.get("inside_rh_pct")

    # Default
    action = {"type": "idle", "intensity": 0.0}
    why = "default idle"

    if not energy_ok:
        # conserve energy
        action = {"type": "idle", "intensity": 0.0}
        why = "energy not ok → conserve (idle)"
        return action, why

    # Basic comfort control
    try:
        t_val = float(t) if t is not None else np.nan
        rh_val = float(rh) if rh is not None else np.nan
    except Exception:
        t_val, rh_val = np.nan, np.nan

    if np.isfinite(t_val) and t_val > 30:
        action = {"type": "ventilate", "intensity": min(1.0, (t_val - 30) / 10)}
        why = f"temp {t_val:.1f}C > 30 → ventilate"
    elif np.isfinite(rh_val) and rh_val > 85:
        action = {"type": "ventilate", "intensity": min(1.0, (rh_val - 85) / 15)}
        why = f"RH {rh_val:.1f}% > 85 → ventilate"
    elif np.isfinite(t_val) and t_val < 12:
        action = {"type": "heat", "intensity": min(1.0, (12 - t_val) / 10)}
        why = f"temp {t_val:.1f}C < 12 → heat"
    else:
        why = "within comfort band → idle"

    return action, why


def compute_outcome(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight outcome proxy: risk_score from thresholds.
    (Later: predicted vs actual, risk_flags from perception, etc.)
    """
    t = state.get("inside_temp_c")
    rh = state.get("inside_rh_pct")

    score = 0
    flags = []

    try:
        t_val = float(t) if t is not None else np.nan
        if np.isfinite(t_val) and (t_val < 10 or t_val > 35):
            score += 1
            flags.append("temp_out_of_bounds")
    except Exception:
        pass

    try:
        rh_val = float(rh) if rh is not None else np.nan
        if np.isfinite(rh_val) and (rh_val < 30 or rh_val > 90):
            score += 1
            flags.append("rh_out_of_bounds")
    except Exception:
        pass

    return {"risk_score": score, "rule_flags": flags}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--artifacts", default="artifacts/day11", type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--max_steps", default=300, type=int, help="limit for quick runs")
    args = ap.parse_args()

    np.random.seed(args.seed)

    out_dir = ensure_dir(Path(args.artifacts))
    df = read_dataset(Path(args.dataset))

    schema_ok, schema_txt, schema_summary = check_schema(df)
    (out_dir / "schema_check.txt").write_text(schema_txt, encoding="utf-8")

    # column mapping (tolerant)
    colmap = {
        "timestamp": pick_col(df, ["ts", "timestamp", "datetime", "time"]),
        "inside_temp_c": pick_col(df, ["inside_temp_c", "temp_c", "t_inside_c", "temperature_c"]),
        "inside_rh_pct": pick_col(df, ["inside_rh_pct", "rh_pct", "humidity_pct"]),
        "light_lux": pick_col(df, ["light_lux", "inside_light_lux"]),
        "energy_available_wh": pick_col(df, ["energy_available_wh", "energy_wh"]),
        "energy_ok": pick_col(df, ["energy_ok"]),
    }

    run_id = f"day11_{utc_now_iso()}"
    logger = SystemLogger(out_dir=out_dir, run_id=run_id)

    # mini-loop
    n = min(len(df), args.max_steps)
    for i in range(n):
        row = df.iloc[i]
        state = build_state(row, colmap)

        # constraints
        energy_ok = True
        if colmap["energy_ok"] is not None:
            val = row.get(colmap["energy_ok"])
            # normalize to bool-ish
            if isinstance(val, str):
                energy_ok = val.strip().lower() in ("1", "true", "yes", "ok")
            else:
                try:
                    energy_ok = bool(int(val))
                except Exception:
                    energy_ok = bool(val)

        constraints = {"energy_ok": energy_ok}

        action, why = decide_action(state, energy_ok)
        outcome = compute_outcome(state)

        logger.log_step(
            step_idx=i,
            state=state,
            action=action,
            constraints=constraints,
            outcome=outcome,
            why=why,
        )

    logger.close()

    # metrics summary
    metrics_path = write_metrics_summary(out_dir)

    # run manifest
    write_json(
        out_dir / "run_manifest.json",
        {
            "day": "day11",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "seed": args.seed,
            "dataset": args.dataset,
            "schema_ok": bool(schema_ok),
            "schema_summary": schema_summary,
            "colmap": colmap,
            "env": env_fingerprint(),
            "max_steps": int(args.max_steps),
        },
    )

    print(f"[Day11] Done. Artifacts at: {out_dir}")
    print(f"[Day11] Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
