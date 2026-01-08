from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.system.utils_io import ensure_dir, write_json, utc_now_iso, env_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day12", default="artifacts/day12/actions_trace.csv", type=str)
    ap.add_argument("--day13", default="artifacts/day13/state_timeline.csv", type=str)
    ap.add_argument("--out_dir", default="artifacts/day13", type=str)
    ap.add_argument("--check_ts", action="store_true", help="Warn if ts mismatches after join")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    p12 = Path(args.day12)
    p13 = Path(args.day13)

    df12 = pd.read_csv(p12)
    df13 = pd.read_csv(p13)

    # Minimal required columns
    for col in ["step_idx"]:
        if col not in df12.columns:
            raise ValueError(f"Missing '{col}' in {p12}")
        if col not in df13.columns:
            raise ValueError(f"Missing '{col}' in {p13}")

    # Keep only what we need from each side (clean + stable)
    df12_keep = df12[
        [
            "step_idx",
            "vent_on",
            "vent_intensity",
            "irrigate",
            "irrigate_intensity",
            "why",
        ]
    ].copy()

    df13_keep = df13[
        [
            "step_idx",
            "ts",
            "state_code",
            "temp_state",
            "soil_state",
            "humid_state",
            "energy_state",
            "explanation",
        ]
    ].copy()

    # Join: one row per timestep
    merged = df13_keep.merge(df12_keep, on="step_idx", how="left", validate="one_to_one")

    # Optional timestamp consistency check
    ts_mismatches = 0
    if args.check_ts and "ts" in df12.columns:
        # compare ts from day12 vs ts from day13 when both exist
        t12 = df12[["step_idx", "ts"]].rename(columns={"ts": "ts_day12"})
        merged2 = merged.merge(t12, on="step_idx", how="left")
        ts_mismatches = int((merged2["ts"].astype(str) != merged2["ts_day12"].astype(str)).sum())
        merged = merged2.drop(columns=["ts_day12"])

    out_csv = out_dir / "state_action_timeline.csv"
    merged.to_csv(out_csv, index=False)

    # Quick summary per state_code (optional but super useful)
    # vent_on / irrigate may come as strings -> normalize to bool-ish
    def _to_bool_series(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "ok"])

    merged["vent_on_bool"] = _to_bool_series(merged["vent_on"])
    merged["irrigate_bool"] = _to_bool_series(merged["irrigate"])

    summary = (
        merged.groupby("state_code")
        .agg(
            count=("step_idx", "count"),
            vent_rate=("vent_on_bool", "mean"),
            irrigate_rate=("irrigate_bool", "mean"),
            vent_intensity_mean=("vent_intensity", "mean"),
            irrigate_intensity_mean=("irrigate_intensity", "mean"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )

    out_summary = out_dir / "state_action_summary.json"
    write_json(
        out_summary,
        {
            "created_utc": utc_now_iso(),
            "rows": int(len(merged)),
            "unique_states": int(merged["state_code"].nunique()),
            "ts_mismatches_checked": bool(args.check_ts),
            "ts_mismatches": int(ts_mismatches),
            "top_states": summary.head(12).to_dict(orient="records"),
            "env": env_fingerprint(),
        },
    )

    print(f"[Day13-Join] Wrote: {out_csv}")
    print(f"[Day13-Join] Wrote summary: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
