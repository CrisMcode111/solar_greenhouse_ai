from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.system.utils_io import ensure_dir, read_dataset, write_json, write_text, utc_now_iso, env_fingerprint
from src.system.schema import check_schema, DEFAULT_SCHEMA


@dataclass
class Day10Config:
    dataset_path: Path
    artifacts_dir: Path
    seed: int = 42
    # dacă vrei, poți adăuga date_range, sample, etc.


def try_run_day09_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    1) încearcă să importe un baseline real din Day09 (dacă îl ai ca funcție).
    2) fallback: calculează un baseline simplu (ex: scor de risc din praguri).
    """
    # --- (A) Try import your actual Day09 baseline ---
    for mod, fn in [
        ("src.ml.day09_baseline", "run_day09_baseline"),
        ("src.system.day09_baseline", "run_day09_baseline"),
        ("src.ml.day09", "run_baseline"),
    ]:
        try:
            m = __import__(mod, fromlist=[fn])
            f = getattr(m, fn)
            out = f(df)
            return {"mode": "day09_imported", "module": mod, "function": fn, "output": out}
        except Exception:
            pass

    # --- (B) Fallback baseline (robust & deterministic) ---
    # Praguri “de bun simț” doar ca să ai ceva măsurabil în Day10.
    # Vei înlocui cu Day09 real.
    temp = pd.to_numeric(df.get("temp_c", pd.Series([np.nan] * len(df))), errors="coerce")
    rh = pd.to_numeric(df.get("rh_pct", pd.Series([np.nan] * len(df))), errors="coerce")
    light = pd.to_numeric(df.get("light_lux", pd.Series([np.nan] * len(df))), errors="coerce")

    risk_temp = (temp < 10) | (temp > 35)
    risk_rh = (rh < 30) | (rh > 85)
    risk_light = (light < 200)  # exemplu

    risk_score = (risk_temp.astype(int) + risk_rh.astype(int) + risk_light.astype(int))
    df_out = df.copy()
    df_out["baseline_risk_score"] = risk_score

    summary = {
        "mode": "fallback_thresholds",
        "risk_score_mean": float(np.nanmean(risk_score)),
        "risk_score_p95": float(np.nanpercentile(risk_score, 95)),
        "risk_rows_ge2": int((risk_score >= 2).sum()),
    }
    return {"mode": "fallback_thresholds", "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to Day08 dataset (.csv/.parquet)")
    ap.add_argument("--artifacts", type=str, default="artifacts/day10", help="Artifacts output dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Day10Config(
        dataset_path=Path(args.dataset),
        artifacts_dir=Path(args.artifacts),
        seed=int(args.seed),
    )

    np.random.seed(cfg.seed)

    out_dir = ensure_dir(cfg.artifacts_dir)

    manifest: Dict[str, Any] = {
        "day": "day10",
        "created_utc": utc_now_iso(),
        "seed": cfg.seed,
        "dataset_path": str(cfg.dataset_path),
        "artifacts_dir": str(out_dir),
        "env": env_fingerprint(),
    }

    # 1) Load dataset
    df = read_dataset(cfg.dataset_path)
    manifest["dataset_rows"] = int(len(df))
    manifest["dataset_cols"] = int(df.shape[1])

    # 2) Schema validation
    schema_ok, schema_txt, schema_summary = check_schema(df, DEFAULT_SCHEMA)
    write_text(out_dir / "schema_check.txt", schema_txt)

    # 3) Baseline step (Day09 imported OR fallback)
    baseline = try_run_day09_baseline(df)

    # 4) Build system report
    system_report: Dict[str, Any] = {
        "day": "day10",
        "created_utc": utc_now_iso(),
        "schema": {
            "ok": bool(schema_ok),
            "summary": schema_summary,
        },
        "baseline": baseline,
        "notes": [
            "Day10 goal: single entrypoint + reproducible artifacts.",
            "Replace fallback baseline with Day09 real baseline when available.",
        ],
    }

    # 5) Write artifacts
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "system_report.json", system_report)

    # Optional: quick human-readable line
    write_text(
        out_dir / "README.txt",
        f"Day10 run complete.\nSchema ok: {schema_ok}\nArtifacts: {out_dir}\n"
    )

    print(f"[Day10] Done. Artifacts at: {out_dir}")
    print(f"[Day10] Schema ok: {schema_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
