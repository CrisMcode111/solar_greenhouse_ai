
# Energy-aware training scheduler.

# Day04: energy budget & frugal scheduling
#Day05: safety and cooldown guards for edge stability



# src/energy/day04_energy_budget_scheduler.py

from __future__ import annotations

import os
import math
import csv
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import src.energy.day03_solar_window_training as day03


# ---------------------------
# Config
# ---------------------------

ARTIFACTS_DIR = Path("artifacts/day04")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV_PATH = ARTIFACTS_DIR / "energy_budget_log.csv"
PLOT_SPENT_VS_ACC = ARTIFACTS_DIR / "energy_spent_vs_val_acc.png"
PLOT_SEASON_COMP = ARTIFACTS_DIR / "season_comparison.png"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model_day04.keras"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model_day04.keras"


@dataclass
class SeasonProfile:
    name: str
    sunrise: float          # hour, e.g. 8.5
    sunset: float           # hour, e.g. 16.5
    cloud_strength: float   # higher = more clouds (more variability, lower peak)


WINTER = SeasonProfile(name="winter", sunrise=8.5, sunset=16.5, cloud_strength=0.55)
SUMMER = SeasonProfile(name="summer", sunrise=6.0, sunset=21.0, cloud_strength=0.25)


@dataclass
class BudgetConfig:
    daily_budget_units: int = 1000
    cost_per_step: int = 1
    max_units_per_hour: int = 140  # scales E -> units/hour
    min_steps_to_train: int = 10

    # frugal policy
    stagnation_hours: int = 4
    stagnation_delta: float = 0.002  # val_acc improvement threshold
    low_budget_threshold: int = 120   # if remaining < this, be conservative
    reduce_steps_factor: float = 0.5  # if stagnation: cut steps


# ---------------------------
# Reuse from Day03 (TODO: adapt imports)
# ---------------------------

import src.energy.day03_solar_window_training as day03


def load_day03_datasets(data_dir: str):
    out = day03.build_datasets_from_directory(
        Path(data_dir),
        img_size=(128, 128),
        batch_size=16,
        val_split=0.2,
        seed=123,
    )

    # Be flexible with Day03 return signature
    # 1) (train_ds, val_ds, class_names)
    # 2) (train_ds, val_ds, num_classes)
    train_ds, val_ds, third = out

    if isinstance(third, (list, tuple)):
        num_classes = len(third)
    else:
        num_classes = int(third)

    return train_ds, val_ds, num_classes



def build_day03_model(num_classes: int):
    return day03.build_mobilenetv2_model(
        num_classes=num_classes,
        img_size=(128, 128),
        lr=1e-3
    )



# ---------------------------
# Energy generator (multi-day + seasonal)
# ---------------------------

def _solar_curve(hour: float, sunrise: float, sunset: float) -> float:
    """Smooth half-sine between sunrise and sunset, else 0."""
    if hour < sunrise or hour > sunset:
        return 0.0
    # map hour -> [0, pi]
    x = (hour - sunrise) / max(1e-6, (sunset - sunrise))
    return math.sin(math.pi * x)  # peak 1 at mid-day


def generate_hourly_energy(
    profile: SeasonProfile,
    day_index: int,
    hours: List[int],
    base_seed: int = 42,
) -> List[float]:
    """
    For each hour: E in [0..1] with clouds.
    Different seed per day -> different clouds.
    """
    rng = np.random.default_rng(base_seed + day_index * 1000 + (0 if profile.name == "winter" else 1))
    energies = []
    for h in hours:
        # clear-sky solar
        clear = _solar_curve(float(h), profile.sunrise, profile.sunset)

        # clouds: multiplicative noise
        # stronger clouds -> lower mean and more dips
        cloud = 1.0 - profile.cloud_strength * rng.random()  # in [1-cloud_strength, 1]
        # occasional extra dips
        if rng.random() < (0.15 + profile.cloud_strength * 0.25):
            cloud *= (0.35 + 0.4 * rng.random())

        E = float(np.clip(clear * cloud, 0.0, 1.0))
        energies.append(E)
    return energies


# ---------------------------
# Budget mechanics
# ---------------------------

def energy_to_available_units(E: float, cfg: BudgetConfig) -> int:
    return int(round(E * cfg.max_units_per_hour))


def steps_possible(available_units: int, budget_remaining: int, cfg: BudgetConfig) -> int:
    usable = min(available_units, budget_remaining)
    return int(usable // cfg.cost_per_step)


# ---------------------------
# Frugal policy helpers
# ---------------------------

def detect_stagnation(val_acc_history: List[float], cfg: BudgetConfig) -> bool:
    if len(val_acc_history) < cfg.stagnation_hours + 1:
        return False
    recent = val_acc_history[-(cfg.stagnation_hours + 1):]
    improvement = max(recent) - min(recent)
    return improvement < cfg.stagnation_delta


# ---------------------------
# Training loop
# ---------------------------

def ensure_csv_header(path: Path):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "season",
            "day",
            "hour",
            "E",
            "energy_available_units",
            "budget_remaining_before",
            "budget_remaining_after",
            "steps_possible",
            "steps_run",
            "trained_bool",
            "val_loss",
            "val_acc",
            "best_val_acc_so_far",
            "spent_cumulative_units"
            "cooldown_remaining",
            "stagnation_count",
            "blocked_reason"

        ])


def evaluate(model: tf.keras.Model, val_ds):
    metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    # be tolerant: accuracy key could be 'accuracy' or 'sparse_categorical_accuracy'
    val_loss = float(metrics.get("loss"))
    val_acc = float(metrics.get("accuracy", metrics.get("sparse_categorical_accuracy", 0.0)))
    return val_loss, val_acc


def train_for_steps(model: tf.keras.Model, train_ds, steps: int):
    # one "hour slice": fit with steps_per_epoch=steps for 1 epoch
    hist = model.fit(train_ds, epochs=1, steps_per_epoch=steps, verbose=0)
    return hist


def run_season(
    profile: SeasonProfile,
    days: int,
    cfg: BudgetConfig,
    data_dir: str,
    base_seed: int,
    cooldown_hours: int,
) -> Dict[str, List[float]]:
    # load once
    train_ds, val_ds, num_classes = load_day03_datasets(data_dir)
    model = build_day03_model(num_classes)

    ensure_csv_header(LOG_CSV_PATH)

    best_val_acc = -1.0
    cooldown_remaining = 0      # hours left before training is allowed again

    stagnation_count = 0        # tracks accuracy stagnation to avoid wasting energy on non-improving training
    last_improve_acc = -1.0     # reference accuracy for detecting meaningful progress



    spent_cumulative = 0



    hours = list(range(24))
    timeline_val_acc = []
    timeline_spent = []

    cooldown_remaining = 0


    for day in range(days):
        budget_remaining = cfg.daily_budget_units
        energies = generate_hourly_energy(profile, day_index=day, hours=hours, base_seed=base_seed)

    for h, E in zip(hours, energies):
        blocked_reason = ""


        # --- Day05: safety guard (stagnation) ---
        if stagnation_count >= args.patience_hours:
            trained = False
            steps_run = 0
            blocked_reason = "stagnation_safety"

        else:
            # --- cooldown tick ---
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # --- cooldown guard ---
            if cooldown_remaining > 0:
                trained = False
                steps_run = 0
                blocked_reason = "cooldown"

            else:
                # --- existing energy / budget logic (Day04) ---
                if sp >= cfg.min_steps_to_train and budget_remaining > 0 and available_units > 0:
                    steps_run = min(sp, budget_remaining // cfg.cost_per_step)
                    if steps_run >= cfg.min_steps_to_train:
                        _ = train_for_steps(model, train_ds, steps=steps_run)
                        trained = True
                        spent_now = steps_run * cfg.cost_per_step
                        budget_remaining -= spent_now
                        spent_cumulative += spent_now
                    else:
                        trained = False
                        steps_run = 0
                else:
                    trained = False
                    steps_run = 0

        if trained:
            cooldown_remaining = args.cooldown_hours
           

        available_units = energy_to_available_units(E, cfg)

        before = budget_remaining
        sp = steps_possible(available_units, budget_remaining, cfg)

        steps_run = 0
        trained = False

        # frugal modifiers (existing)
        stagnating = detect_stagnation(timeline_val_acc, cfg)
        if stagnating and sp > 0:
            sp = max(cfg.min_steps_to_train, int(sp * cfg.reduce_steps_factor))

        if budget_remaining < cfg.low_budget_threshold:
            sp = min(sp, cfg.min_steps_to_train)

         
        steps_run = 0

        # start cooldown AFTER training
        if trained:
            cooldown_remaining = cooldown_hours

        # always evaluate
        val_loss, val_acc = evaluate(model, val_ds)


        # --- Day05: safety guard against accuracy stagnation ---
        if val_acc > last_improve_acc + args.min_acc_improve:
            last_improve_acc = val_acc
            stagnation_count = 0
        else:
            stagnation_count += 1

        # checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(BEST_MODEL_PATH)

        timeline_val_acc.append(val_acc)
        timeline_spent.append(spent_cumulative)

        # log
        with LOG_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                int(time.time()),
                profile.name,
                day,
                h,
                round(float(E), 6),
                int(available_units),
                int(before),
                int(budget_remaining),
                int(sp),
                int(steps_run),
                1 if trained else 0,
                round(float(val_loss), 6),
                round(float(val_acc), 6),
                round(float(best_val_acc), 6),
                int(spent_cumulative),
                int(cooldown_remaining),
                0,                 # stagnation_count (Day05 next)
                blocked_reason
            ])

# final save (after all days)
    model.save(FINAL_MODEL_PATH)

    return {
        "val_acc": timeline_val_acc,
        "spent": timeline_spent
        }



# ---------------------------
# Plotting
# ---------------------------

def plot_spent_vs_acc(spent: List[float], acc: List[float], outpath: Path, title: str):
    plt.figure()
    plt.plot(spent, acc)
    plt.xlabel("Cumulative energy spent (units)")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_season_comparison(
    winter: Dict[str, List[float]],
    summer: Dict[str, List[float]],
    outpath: Path
):
    plt.figure()
    plt.plot(winter["val_acc"], label="winter val_acc")
    plt.plot(summer["val_acc"], label="summer val_acc")
    plt.xlabel("Hour index (across days)")
    plt.ylabel("Validation accuracy")
    plt.title("Season comparison: validation accuracy over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dataset_greenhouse")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

# Day05: safety & cooldown params 
    parser.add_argument(
    "--cooldown_hours",
    type=int,
    default=1,
    help="Number of hours to wait after a training step before allowing training again (anti-thrashing)"
    )

    parser.add_argument(
    "--patience_hours",
    type=int,
    default=3,
    help="Number of consecutive hours without validation improvement before triggering safety block"
    )

    parser.add_argument(
    "--min_acc_improve",
    type=float,
    default=0.002,
    help="Minimum validation accuracy improvement to be considered meaningful"
   )

    # budget params
    parser.add_argument("--daily_budget_units", type=int, default=1000)
    parser.add_argument("--cost_per_step", type=int, default=1)
    parser.add_argument("--max_units_per_hour", type=int, default=140)
    parser.add_argument("--min_steps_to_train", type=int, default=10)

    args = parser.parse_args()

    cfg = BudgetConfig(
        daily_budget_units=args.daily_budget_units,
        cost_per_step=args.cost_per_step,
        max_units_per_hour=args.max_units_per_hour,
        min_steps_to_train=args.min_steps_to_train,
    )

    # fresh csv each run (optional)
    # if LOG_CSV_PATH.exists():
    # LOG_CSV_PATH.unlink()


    winter_res = run_season(
        WINTER,
        days=args.days,
        cfg=cfg,
        data_dir=args.data_dir,
        base_seed=args.seed,
        cooldown_hours=args.cooldown_hours,
   )

    summer_res = run_season(
        SUMMER,
        days=args.days,
        cfg=cfg,
        data_dir=args.data_dir,
        base_seed=args.seed,
        cooldown_hours=args.cooldown_hours,
    )


    # plot 1: spent vs acc (use winter by default)
    plot_spent_vs_acc(
        spent=winter_res["spent"],
        acc=winter_res["val_acc"],
        outpath=PLOT_SPENT_VS_ACC,
        title="Energy spent vs validation accuracy (winter)"
    )

    # plot 2: season comparison
    plot_season_comparison(winter_res, summer_res, PLOT_SEASON_COMP)

    print(f"[OK] Artifacts saved in: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
