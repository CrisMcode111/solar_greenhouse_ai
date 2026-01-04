from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    in_path = Path("artifacts/day08/synthetic_sensors_energy_coupled.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    out_dir = Path("artifacts/day08/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Parse timestamp safely (keeps timezone if present, otherwise naive)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # --- Plot 1: energy_available_wh over time ---
    if "energy_available_wh" in df.columns:
        plt.figure()
        plt.plot(df["timestamp"], df["energy_available_wh"])
        plt.title("Energy Available (Wh) over Time")
        plt.xlabel("Time")
        plt.ylabel("energy_available_wh")
        plt.tight_layout()
        plt.savefig(out_dir / "01_energy_available_wh.png", dpi=160)
        plt.close()

    # --- Plot 2: energy_ok over time (step plot) ---
    if "energy_ok" in df.columns:
        plt.figure()
        plt.step(df["timestamp"], df["energy_ok"], where="post")
        plt.title("energy_ok over Time")
        plt.xlabel("Time")
        plt.ylabel("energy_ok")
        plt.yticks([0, 1])
        plt.tight_layout()
        plt.savefig(out_dir / "02_energy_ok.png", dpi=160)
        plt.close()

    # --- Plot 3: inside_temp_c with energy_ok overlay (mask) ---
    if "inside_temp_c" in df.columns and "energy_ok" in df.columns:
        plt.figure()
        plt.plot(df["timestamp"], df["inside_temp_c"], label="inside_temp_c")

        # Overlay points where energy_ok == 1 (no explicit color, default)
        mask = df["energy_ok"].astype(float) == 1.0
        plt.scatter(df.loc[mask, "timestamp"], df.loc[mask, "inside_temp_c"], s=10, label="energy_ok=1")

        plt.title("Inside Temperature with energy_ok Overlay")
        plt.xlabel("Time")
        plt.ylabel("inside_temp_c")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "03_inside_temp_with_energy_ok.png", dpi=160)
        plt.close()

    # --- Plot 4: daily summary (mean energy, % energy_ok) ---
    if "energy_available_wh" in df.columns and "energy_ok" in df.columns:
        daily = df.set_index("timestamp").resample("1D").agg(
            mean_energy_wh=("energy_available_wh", "mean"),
            pct_energy_ok=("energy_ok", "mean"),
        ).dropna()

        plt.figure()
        plt.plot(daily.index, daily["mean_energy_wh"])
        plt.title("Daily Mean Energy Available (Wh)")
        plt.xlabel("Day")
        plt.ylabel("mean_energy_wh")
        plt.tight_layout()
        plt.savefig(out_dir / "04_daily_mean_energy_wh.png", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(daily.index, daily["pct_energy_ok"])
        plt.title("Daily % energy_ok (mean of energy_ok)")
        plt.xlabel("Day")
        plt.ylabel("pct_energy_ok")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / "05_daily_pct_energy_ok.png", dpi=160)
        plt.close()

    print("[OK] Saved plots to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
