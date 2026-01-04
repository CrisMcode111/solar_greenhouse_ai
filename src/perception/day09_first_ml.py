from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err * err)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> int:
    in_path = Path("artifacts/day08/synthetic_sensors_energy_coupled.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    out_dir = Path("artifacts/day09")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Target: inside_temp at t+1
    df["y_inside_temp_next"] = df["inside_temp_c"].shift(-1)

    # Basic feature set (t)
    feature_cols = [
        "inside_temp_c",
        "outside_temp_c",
        "light_lux",
        "inside_rh_pct",
        "outside_rh_pct",
        "soil_moisture_pct",
        "vent_state",
        "energy_available_wh",
        "energy_ok",
    ]

    # Drop rows with missing in features/target (noise introduces NaNs)
    data = df[["timestamp"] + feature_cols + ["y_inside_temp_next"]].dropna().copy()

    # Time-based split (70% train, 30% test)
    n = len(data)
    split = int(n * 0.7)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    # Energy-aware training gate: train only where energy_ok == 1
    train_gated = train[train["energy_ok"] == 1].copy()
    if len(train_gated) < 50:
        print("[WARN] Very few gated training samples. Check energy_ok logic or date range.")

    X_train = train_gated[feature_cols].to_numpy(dtype=float)
    y_train = train_gated["y_inside_temp_next"].to_numpy(dtype=float)

    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["y_inside_temp_next"].to_numpy(dtype=float)

    # Try sklearn first; fallback to closed-form linear regression
    y_pred = None
    model_type = None
    coefs = None
    intercept = None

    try:
        from sklearn.linear_model import LinearRegression  # type: ignore

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_type = "sklearn.LinearRegression"
        coefs = model.coef_.tolist()
        intercept = float(model.intercept_)

    except Exception:
        # Closed-form OLS with intercept: beta = (X'X)^-1 X'y
        Xb = np.concatenate([np.ones((len(X_train), 1)), X_train], axis=1)
        XtX = Xb.T @ Xb
        Xty = Xb.T @ y_train
        beta = np.linalg.pinv(XtX) @ Xty  # stable pseudo-inverse
        intercept = float(beta[0])
        w = beta[1:]
        coefs = w.tolist()
        y_pred = (intercept + X_test @ w)
        model_type = "numpy.OLS_pinv"

    # Metrics (overall test)
    metrics = {
        "model": model_type,
        "n_total": int(n),
        "n_train": int(len(train)),
        "n_train_gated_energy_ok_1": int(len(train_gated)),
        "n_test": int(len(test)),
        "test_mae": _mae(y_test, y_pred),
        "test_rmse": _rmse(y_test, y_pred),
    }

    # Metrics on energy_ok==1 subset in test (optional but insightful)
    test_energy1 = test[test["energy_ok"] == 1].copy()
    if len(test_energy1) > 0:
        X_test_e1 = test_energy1[feature_cols].to_numpy(dtype=float)
        y_test_e1 = test_energy1["y_inside_temp_next"].to_numpy(dtype=float)
        y_pred_e1 = None
        if model_type.startswith("sklearn"):
            y_pred_e1 = model.predict(X_test_e1)  # type: ignore
        else:
            y_pred_e1 = intercept + X_test_e1 @ np.array(coefs, dtype=float)
        metrics["test_energy_ok_1_mae"] = _mae(y_test_e1, y_pred_e1)
        metrics["test_energy_ok_1_rmse"] = _rmse(y_test_e1, y_pred_e1)
        metrics["n_test_energy_ok_1"] = int(len(test_energy1))

    # Save metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save coefficients
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
    })
    coef_df.loc[len(coef_df)] = ["intercept", intercept]
    coef_df.to_csv(out_dir / "model_coefficients.csv", index=False)

    # Save predictions
    pred_df = test[["timestamp"] + feature_cols + ["y_inside_temp_next"]].copy()
    pred_df["y_pred"] = y_pred
    pred_df["error"] = pred_df["y_inside_temp_next"] - pred_df["y_pred"]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Plot 1: Pred vs actual (time)
    plt.figure()
    plt.plot(pred_df["timestamp"], pred_df["y_inside_temp_next"], label="actual")
    plt.plot(pred_df["timestamp"], pred_df["y_pred"], label="pred")
    plt.title("Day 09 – Inside Temp (t+1) Prediction (Test)")
    plt.xlabel("Time")
    plt.ylabel("inside_temp_c(t+1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "01_pred_vs_actual.png", dpi=160)
    plt.close()

    # Plot 2: Error over time
    plt.figure()
    plt.plot(pred_df["timestamp"], pred_df["error"])
    plt.title("Day 09 – Prediction Error over Time (Test)")
    plt.xlabel("Time")
    plt.ylabel("error = actual - pred")
    plt.tight_layout()
    plt.savefig(plots_dir / "02_error_over_time.png", dpi=160)
    plt.close()

    print("[OK] Day 09 artifacts saved to:", out_dir)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
