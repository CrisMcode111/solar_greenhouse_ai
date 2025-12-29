# ======================================================
# Day 03 – Solar Window Training
# Imports & Sanity Checks
# ======================================================

import os
import sys
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- TensorFlow (CPU) ---
# Reduce TF logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import backend as K


# ======================================================
# Reproducibility
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ======================================================
# Sanity checks (VERY IMPORTANT)
# ======================================================
def sanity_checks():
    print("=" * 60)
    print("Day 03 – Solar Window Training : Sanity Checks")
    print("- Python executable :", sys.executable)
    print("- Python version    :", sys.version.split()[0])
    print("- TensorFlow version:", tf.__version__)
    print("- GPUs detected     :", tf.config.list_physical_devices("GPU"))
    print("- NumPy test        :", np.array([1, 2, 3]).sum())
    print("=" * 60)


if __name__ == "__main__":
    # Clear previous TF graphs / sessions
    K.clear_session()
    gc.collect()

    sanity_checks()


# Paths & Artifacts (local, VS Code)

def get_paths():
    """
    Resolve project paths in a robust way, regardless of where you run the script from.
    Assumes script location: <PROJECT_ROOT>/src/energy/day03_solar_window_training.py
    """
    project_root = Path(__file__).resolve().parents[2]  # .../solar_greenhouse_ai

    # Dataset folders (NOT tracked in git)
    data_root = project_root / "data"

    # IMPORTANT: set this to your real folder name inside /data
    # If your folder is still called "color", change to: data_root / "color"
    plantvillage_dir = data_root / "plantvillage_color"

    # Filtered subset (generated locally)
    filtered_dir = data_root / "dataset_greenhouse"

    # Output artifacts (generated locally)
    artifacts_dir = project_root / "artifacts" / "day03"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "sera_solar_day03.keras"
    history_csv = artifacts_dir / "sera_solar_day03_history.csv"
    classes_json = artifacts_dir / "class_names.json"

    return {
        "PROJECT_ROOT": project_root,
        "DATA_ROOT": data_root,
        "PLANTVILLAGE_DIR": plantvillage_dir,
        "FILTERED_DIR": filtered_dir,
        "ARTIFACTS_DIR": artifacts_dir,
        "MODEL_PATH": model_path,
        "HIST_CSV": history_csv,
        "CLASSES_JSON": classes_json,
    }


def print_paths(paths: dict):
    print("\n" + "=" * 60)
    print("Paths")
    for k, v in paths.items():
        print(f"- {k}: {v}")
    print("=" * 60 + "\n")


def validate_paths(paths: dict):
    # dataset must exist
    if not paths["PLANTVILLAGE_DIR"].exists():
        raise FileNotFoundError(
            f"PlantVillage folder not found:\n  {paths['PLANTVILLAGE_DIR']}\n\n"
            "Fix: put the PlantVillage *color* folder in:\n"
            f"  {paths['DATA_ROOT']}\n"
            "and name it 'plantvillage_color' (or update PLANTVILLAGE_DIR in get_paths())."
        )

    # data root should exist too
    paths["DATA_ROOT"].mkdir(parents=True, exist_ok=True)

    return True


if __name__ == "__main__":
    # ... existing sanity checks already ran above ...
    paths = get_paths()
    print_paths(paths)
    validate_paths(paths)
    print(" Paths OK. Ready for next steps.")

