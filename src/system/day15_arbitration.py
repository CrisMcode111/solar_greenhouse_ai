from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.system.utils_io import ensure_dir, write_json, utc_now_iso, env_fingerprint
from src.system.policy_v1 import policy_decide_from_state, PolicyConfigV1
from src.system.arbitrator import arbitrate, ArbitrationConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Day13 state_action_timeline.csv")
    ap.add_argument("--out_dir", default="artifacts/day15", type=str)
    ap.add_argument("--max_rows", default=0, type=int, help="0 = all rows")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    df = pd.read_csv(Path(args.input))

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    run_id = f"day15_{utc_now_iso()}"

    # Config snapshots
    policy_cfg = PolicyConfigV1()
    arb_cfg = ArbitrationConfig()

    write_json(out_dir / "arbitration_config.json", {
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "policy_config": policy_cfg.__dict__,
        "arbitration_config": arb_cfg.__dict__,
        "env": env_fingerprint(),
        "input": args.input,
        "rows": int(len(df)),
    })

    final_rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        state_code = str(r["state_code"])
        step_idx = int(r["step_idx"])
        ts = r.get("ts")

        # Rules action comes from Day12 trace already merged into Day13
        rules_action = {
            "vent_on": bool(r.get("vent_on")),
            "vent_intensity": float(r.get("vent_intensity") or 0.0),
            "irrigate": bool(r.get("irrigate")),
            "irrigate_intensity": float(r.get("irrigate_intensity") or 0.0),
        }
        rules_why = str(r.get("why") or "")

        # Policy action computed from state_code
        policy_action, policy_why = policy_decide_from_state(state_code, policy_cfg)

        # Arbitration
        final_action, final_why, meta = arbitrate(
            state_code=state_code,
            rules_action=rules_action,
            rules_why=rules_why,
            policy_action=policy_action,
            policy_why=policy_why,
            cfg=arb_cfg,
        )

        # Events: only log interesting things
        if meta.get("override"):
            events.append({
                "run_id": run_id,
                "step_idx": step_idx,
                "ts": ts,
                "state_code": state_code,
                "decision_source": meta.get("decision_source"),
                "final_why": final_why,
                "stress": meta.get("stress"),
            })

        final_rows.append({
            "run_id": run_id,
            "step_idx": step_idx,
            "ts": ts,
            "state_code": state_code,

            # Rules
            "rules_vent_on": rules_action["vent_on"],
            "rules_vent_intensity": rules_action["vent_intensity"],
            "rules_irrigate": rules_action["irrigate"],
            "rules_irrigate_intensity": rules_action["irrigate_intensity"],
            "rules_why": rules_why,

            # Policy
            "policy_vent_on": policy_action["vent_on"],
            "policy_vent_intensity": policy_action["vent_intensity"],
            "policy_irrigate": policy_action["irrigate"],
            "policy_irrigate_intensity": policy_action["irrigate_intensity"],
            "policy_why": policy_why,

            # Stress
            "heat_stress": meta["stress"]["heat_stress"],
            "water_stress": meta["stress"]["water_stress"],
            "cold_vent_risk": meta["stress"]["cold_vent_risk"],
            "total_stress": meta["stress"]["total_stress"],

            # Final
            "final_vent_on": final_action["vent_on"],
            "final_vent_intensity": final_action["vent_intensity"],
            "final_irrigate": final_action["irrigate"],
            "final_irrigate_intensity": final_action["irrigate_intensity"],
            "decision_source": meta.get("decision_source"),
            "final_why": final_why,
        })

    # Write outputs
    final_path = out_dir / "final_actions.csv"
    pd.DataFrame(final_rows).to_csv(final_path, index=False)

    events_path = out_dir / "arbitration_log.jsonl"
    with events_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Metrics summary
    total = len(final_rows)
    overrides = sum(1 for x in final_rows if x.get("decision_source") not in ("rules", "keep rules"))
    final_act_rate = float(pd.DataFrame(final_rows)["final_vent_on"].mean() + pd.DataFrame(final_rows)["final_irrigate"].mean()) / 2.0

    summary = {
        "run_id": run_id,
        "rows": int(total),
        "override_rows": int(len(events)),
        "override_rate": float(len(events) / total) if total else 0.0,
        "final_actuation_rate_avg": float(final_act_rate),
        "mean_total_stress": float(pd.DataFrame(final_rows)["total_stress"].mean()) if total else 0.0,
        "decision_source_counts": pd.DataFrame(final_rows)["decision_source"].value_counts().to_dict(),
    }
    write_json(out_dir / "stress_summary.json", summary)

    print(f"[Day15] Done. Artifacts at: {out_dir}")
    print(f"[Day15] Wrote: {final_path.name}, {events_path.name}, stress_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
