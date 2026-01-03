# src/perception/first_ml.py
"""
Phase 2 – Perception
Day 06: First end-to-end smoke test (no heavy ML yet)

Pipeline:
1) generate base sensors (clean)
2) apply noise/corruptions
3) compute rule-based risk flags
4) save dataset to outputs/
"""

from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd

from src.perception.sensors import SensorGenConfig, generate_synthetic_sensors
from src.perception.noise import NoiseConfig, apply_noise
from src.perception.plant_rules import PlantRuleConfig, compute_risk_flags


def main() -> int:
    out_dir = Path("artifacts/day06")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = SensorGenConfig(
        start="2025-10-01",
        end="2025-10-15",
        freq="1H",
        tz="Europe/Paris",
        seed=42,
    )

    try:
        df = generate_synthetic_sensors(gen_cfg)
    except NotImplementedError as e:
        print("\n[STOP] Synthetic sensor generator is not implemented yet.")
        print("Implement it in: src/perception/sensors.py")
        print("Details:", e, "\n")
        return 1

    # Noise (will be NotImplemented for now -> handle gracefully)
    noise_cfg = NoiseConfig(missing_rate=0.02, spike_rate=0.005, drift_per_day=0.02)
    try:
        df_noisy = apply_noise(df, noise_cfg)
    except NotImplementedError:
        print("[INFO] Noise not implemented yet -> continuing with clean data.")
        df_noisy = df

    # Rules
    rules_cfg = PlantRuleConfig()
    df_final = compute_risk_flags(df_noisy, rules_cfg)

    # Minimal sanity checks
    expected = ["timestamp", "inside_temp_c", "outside_temp_c", "light_lux", "energy_ok"]
    missing = [c for c in expected if c not in df_final.columns]
    if missing:
        print("[WARN] Missing expected columns:", missing)

    out_path = out_dir / "phase2_day06_synthetic_sensors_sample.csv"
    df_final.to_csv(out_path, index=False)
    print("[OK] Saved:", out_path)
    print(df_final.head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
