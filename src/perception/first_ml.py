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

from src.perception.sensors import SensorGenConfig, generate_synthetic_sensors
from src.perception.noise import NoiseConfig, apply_noise
from src.perception.plant_rules import PlantRuleConfig, compute_risk_flags
from src.perception.energy_bridge import compute_energy_ok


def main() -> int:
    # Day 08 output
    out_dir = Path("artifacts/day08")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = SensorGenConfig(
        start="2025-10-01",
        end="2025-10-15",
        freq="1h",  # optional: change to "1h" to silence pandas warning
        tz="Europe/Paris",
        seed=42,
    )

    # 1) Generate clean sensors (Day 06)
    df = generate_synthetic_sensors(gen_cfg)

    # 2) Apply noise (Day 07)
    noise_cfg = NoiseConfig(
        missing_rate=0.02,
        spike_rate=0.005,
        drift_per_day_map={
            "inside_temp_c": 0.02,
            "inside_rh_pct": 0.05,
        },
    )
    df_noisy = apply_noise(df, noise_cfg)

    # 3) Apply plant rules
    rules_cfg = PlantRuleConfig()
    df_final = compute_risk_flags(df_noisy, rules_cfg)

    # 4) Day 08 – overwrite energy_ok using frozen Energy logic
    df_final["energy_ok"] = compute_energy_ok(
        df_final["timestamp"],
        df_final["energy_available_wh"],
    )

    # 5) Save Day 08 artifact
    out_path = out_dir / "synthetic_sensors_energy_coupled.csv"
    df_final.to_csv(out_path, index=False)

    print("[OK] Saved:", out_path)
    print(df_final.head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
