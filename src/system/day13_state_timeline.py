from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.system.utils_io import ensure_dir, read_dataset, write_json, utc_now_iso, env_fingerprint
from src.system.schema import check_schema
from src.system.state_mapping import StateMappingConfig, map_state


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="Path to Day08 dataset CSV")
    ap.add_argument("--artifacts", default="artifacts/day13", type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--max_steps", default=500, type=int)
    ap.add_argument("--sample_n", default=50, type=int, help="How many explanations to sample into JSONL")
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = ensure_dir(Path(args.artifacts))
    df = read_dataset(Path(args.dataset))

    # schema check (informational)
    schema_ok, schema_txt, schema_summary = check_schema(df)
    (out_dir / "schema_check.txt").write_text(schema_txt, encoding="utf-8")

    # tolerant column mapping (aligned with Day12 expectations)
    colmap = {
        "timestamp": pick_col(df, ["ts", "timestamp", "datetime", "time"]),
        "inside_temp_c": pick_col(df, ["inside_temp_c", "temp_c", "t_inside_c", "temperature_c"]),
        "inside_rh_pct": pick_col(df, ["inside_rh_pct", "rh_pct", "humidity_pct"]),
        "soil_moisture": pick_col(df, ["soil_moisture", "soil_moisture_norm", "soil_moisture_pct"]),
        "energy_ok": pick_col(df, ["energy_ok"]),
    }

    n = min(len(df), int(args.max_steps))

    # Mapping config aligned with Day12 (temp_hi=30, rh_hi=85, soil_lo=0.42, soil_hi=0.70)
    cfg = StateMappingConfig()

    rows: List[Dict[str, Any]] = []
    state_counts: Dict[str, int] = {}

    for i in range(n):
        row = df.iloc[i]

        def get(name: str):
            c = colmap.get(name)
            return None if c is None else row.get(c)

        raw_state = {
            "inside_temp_c": get("inside_temp_c"),
            "inside_rh_pct": get("inside_rh_pct"),
            "soil_moisture": get("soil_moisture"),
            "energy_ok": get("energy_ok"),
        }

        mapped = map_state(raw_state, cfg)

        state_code = mapped["state_code"]
        state_counts[state_code] = state_counts.get(state_code, 0) + 1

        rows.append(
            {
                "step_idx": i,
                "ts": get("timestamp"),
                "state_code": state_code,
                "temp_state": mapped["temp_state"],
                "soil_state": mapped["soil_state"],
                "humid_state": mapped["humid_state"],
                "energy_state": mapped["energy_state"],
                "soil_moisture_norm": mapped["soil_moisture_norm"],
                "explanation": mapped["explanation"],
            }
        )

    # --- Artifacts ---
    timeline_path = out_dir / "state_timeline.csv"
    dist_path = out_dir / "state_distribution.json"
    sample_path = out_dir / "explanations_sample.jsonl"

    pd.DataFrame(rows).to_csv(timeline_path, index=False)

    # distribution summary
    write_json(
        dist_path,
        {
            "rows": n,
            "unique_states": len(state_counts),
            "states": dict(sorted(state_counts.items(), key=lambda kv: kv[1], reverse=True)),
        },
    )

    # explanations sample
    sample_n = min(int(args.sample_n), len(rows))
    sampled = random.sample(rows, k=sample_n) if sample_n > 0 else []

    with sample_path.open("w", encoding="utf-8") as f:
        for r in sampled:
            f.write(
                json.dumps(
                    {
                        "step_idx": r["step_idx"],
                        "ts": r["ts"],
                        "state_code": r["state_code"],
                        "temp_state": r["temp_state"],
                        "soil_state": r["soil_state"],
                        "humid_state": r["humid_state"],
                        "energy_state": r["energy_state"],
                        "explanation": r["explanation"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # run manifest
    run_id = f"day13_{utc_now_iso()}"
    write_json(
        out_dir / "run_manifest.json",
        {
            "day": "day13",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "seed": args.seed,
            "dataset": args.dataset,
            "schema_ok": bool(schema_ok),
            "schema_summary": schema_summary,
            "colmap": colmap,
            "mapping_config": {
                "temp_lo_c": cfg.temp_lo_c,
                "temp_hi_c": cfg.temp_hi_c,
                "rh_lo_pct": cfg.rh_lo_pct,
                "rh_hi_pct": cfg.rh_hi_pct,
                "soil_lo": cfg.soil_lo,
                "soil_hi": cfg.soil_hi,
            },
            "max_steps": int(args.max_steps),
            "sample_n": int(args.sample_n),
            "env": env_fingerprint(),
        },
    )

    print(f"[Day13] Done. Artifacts at: {out_dir}")
    print(f"[Day13] Wrote: {timeline_path.name}, {dist_path.name}, {sample_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
