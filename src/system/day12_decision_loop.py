from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.system.utils_io import ensure_dir, read_dataset, write_json, utc_now_iso, env_fingerprint
from src.system.schema import check_schema
from src.system.rules_engine import RulesConfig, decide, config_to_json


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_risk_flags(row: pd.Series) -> List[str]:
    """
    Supports two styles:
      1) a column 'rule_flags' containing a list-like string (e.g. "['rh_out_of_bounds']")
      2) boolean flag columns starting with 'risk_' or 'flag_' set to 1/True
    """
    flags: List[str] = []

    if "rule_flags" in row.index and pd.notna(row["rule_flags"]):
        val = row["rule_flags"]
        if isinstance(val, str):
            # very lightweight parse
            s = val.strip()
            # remove brackets
            s = s.strip("[]")
            if s:
                parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
                flags.extend([p for p in parts if p])
        elif isinstance(val, list):
            flags.extend([str(x) for x in val])

    # boolean columns
    for col in row.index:
        if col.startswith("risk_") or col.startswith("flag_"):
            try:
                v = row[col]
                if isinstance(v, str):
                    ok = v.strip().lower() in ("1", "true", "yes", "ok")
                else:
                    ok = bool(int(v))
                if ok:
                    flags.append(col)
            except Exception:
                continue

    # de-dup
    return sorted(list(set(flags)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--artifacts", default="artifacts/day12", type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--max_steps", default=500, type=int)
    args = ap.parse_args()

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
        "soil_moisture": pick_col(df, ["soil_moisture", "soil_moisture_norm", "soil_moisture_pct"]),
    }

    # config (can be extended later via CLI/JSON)
    cfg = RulesConfig()

    # save config
    write_json(out_dir / "rules_config.json", config_to_json(cfg))

    run_id = f"day12_{utc_now_iso()}"

    # output CSV
    trace_path = out_dir / "actions_trace.csv"
    confusion_path = out_dir / "confusion_rules_vs_risks.csv"

    fieldnames = [
        "run_id",
        "step_idx",
        "ts",
        "energy_ok",
        "inside_temp_c",
        "inside_rh_pct",
        "soil_moisture",
        "risk_flags",
        "vent_on",
        "vent_intensity",
        "irrigate",
        "irrigate_intensity",
        "why",
    ]

    n = min(len(df), args.max_steps)

    # For optional sanity check
    confusion_rows: List[Dict[str, Any]] = []

    with trace_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i in range(n):
            row = df.iloc[i]

            def get(name: str):
                c = colmap.get(name)
                return None if c is None else row.get(c)

            state = {
                "timestamp": get("timestamp"),
                "inside_temp_c": get("inside_temp_c"),
                "inside_rh_pct": get("inside_rh_pct"),
                "light_lux": get("light_lux"),
                "energy_available_wh": get("energy_available_wh"),
                "soil_moisture": get("soil_moisture"),
            }

            constraints = {"energy_ok": get("energy_ok")}
            risk_flags = extract_risk_flags(row)

            action, why, debug = decide(state, constraints, risk_flags, cfg)

            w.writerow({
                "run_id": run_id,
                "step_idx": i,
                "ts": state.get("timestamp"),
                "energy_ok": debug.get("energy_ok"),
                "inside_temp_c": debug.get("temp_c"),
                "inside_rh_pct": debug.get("rh_pct"),
                "soil_moisture": debug.get("soil_moisture"),
                "risk_flags": "|".join(risk_flags),
                "vent_on": action["vent_on"],
                "vent_intensity": action["vent_intensity"],
                "irrigate": action["irrigate"],
                "irrigate_intensity": action["irrigate_intensity"],
                "why": why,
            })

            # optional confusion: when risk flags exist but action stayed OFF
            has_risk = len(risk_flags) > 0
            acted = bool(action["vent_on"] or action["irrigate"])
            confusion_rows.append({
                "step_idx": i,
                "has_risk": has_risk,
                "acted": acted,
                "energy_ok": debug.get("energy_ok"),
                "risk_flags": "|".join(risk_flags),
                "why": why,
            })

    # write confusion table
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)

    # run manifest
    write_json(
        out_dir / "run_manifest.json",
        {
            "day": "day12",
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

    print(f"[Day12] Done. Artifacts at: {out_dir}")
    print(f"[Day12] Wrote: {trace_path.name}, {confusion_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
