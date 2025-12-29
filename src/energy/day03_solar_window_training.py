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

# Solar Energy Simulation (Day 03 core)
from datetime import datetime, timedelta

THRESHOLD = 0.6  # energy gate


def build_energy_simulator(
    sunrise: float = 7.0,
    sunset: float = 19.0,
    cloud_strength: float = 0.15,
    smooth_k: int = 5,
    seed: int = 42,
):
    """
    Returns:
      t_hours: np.array of hours [0..24) sampled every 5 minutes
      E_profile: solar energy with clouds (0..1)
      E_ideal: ideal solar curve (0..1)
      energy_at_hour(h): function that returns energy for scalar or array h
    """
    start = datetime(2025, 1, 1, 0, 0, 0)
    minutes = np.arange(0, 24 * 60, 5)
    times = [start + timedelta(minutes=int(m)) for m in minutes]
    t_hours = np.array([t.hour + t.minute / 60 for t in times], dtype=float)

    # Idealized day curve (sinus between sunrise and sunset)
    daylen = sunset - sunrise
    E_ideal = np.zeros_like(t_hours, dtype=float)
    mask_day = (t_hours >= sunrise) & (t_hours <= sunset)
    phase = (t_hours[mask_day] - sunrise) / daylen * np.pi
    E_ideal[mask_day] = np.sin(phase)

    # Clouds: gaussian noise + moving average smoothing
    rng = np.random.default_rng(seed)
    cloud_noise = rng.normal(loc=0.0, scale=cloud_strength, size=E_ideal.shape)

    k = int(max(1, smooth_k))
    kernel = np.ones(k, dtype=float) / k
    cloud_noise_smoothed = np.convolve(cloud_noise, kernel, mode="same")

    E_profile = np.clip(E_ideal + cloud_noise_smoothed, 0.0, 1.0)

    def energy_at_hour(h):
        """
        Energy at hour h (float) or array-like of hours in [0..24).
        Uses linear interpolation on the precomputed profile.
        """
        h_arr = np.atleast_1d(h).astype(float)
        h_arr = np.clip(h_arr, 0.0, 23.999)
        e_arr = np.interp(h_arr, t_hours, E_profile)
        return float(e_arr[0]) if np.isscalar(h) else e_arr

    return t_hours, E_profile, E_ideal, energy_at_hour


def save_energy_plot(t_hours, E_profile, E_ideal, threshold, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 3))
    plt.plot(t_hours, E_profile, label="E_solar (cu nori)")
    plt.plot(t_hours, E_ideal, linestyle=":", label="E_ideal (fără nori)")
    plt.axhline(threshold, linestyle="--", label=f"Prag={threshold}")
    plt.xlim(0, 24)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f" Saved energy plot: {out_path}")


# ======================================================
# Dataset filtering: PlantVillage -> Greenhouse subset
# ======================================================

import shutil
import json


ALLOWED_KEYWORDS = [
    "tomato",
    "pepper",
    "bell_pepper",
    "grape",
    "soybean",
    "bean",
    "strawberry",
    "raspberry",
]


def list_class_folders(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def filter_classes_by_keywords(class_folders, allowed_keywords):
    allowed = []
    for p in class_folders:
        name = p.name.lower()
        if any(k.lower() in name for k in allowed_keywords):
            allowed.append(p)
    return allowed


def copy_filtered_dataset(
    src_root: Path,
    dst_root: Path,
    allowed_keywords=ALLOWED_KEYWORDS,
    exts=(".jpg", ".jpeg", ".png", ".bmp"),
    overwrite: bool = False,
):
    """
    Copies only selected class folders into dst_root.
    Returns a summary dict (counts, class names).
    """
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        else:
            # already exists: we won't recopy; just summarize what is there
            existing_classes = list_class_folders(dst_root)
            total_imgs = 0
            for cls in existing_classes:
                for ext in exts:
                    total_imgs += len(list(cls.glob(f"*{ext}")))
                    total_imgs += len(list(cls.glob(f"*{ext.upper()}")))
            return {
                "mode": "reused_existing_filtered",
                "dst_root": str(dst_root),
                "num_classes": len(existing_classes),
                "classes": [p.name for p in existing_classes],
                "num_images": total_imgs,
            }

    dst_root.mkdir(parents=True, exist_ok=True)

    all_classes = list_class_folders(src_root)
    allowed_class_folders = filter_classes_by_keywords(all_classes, allowed_keywords)

    # Copy
    copied = 0
    for cls_path in allowed_class_folders:
        dst_cls = dst_root / cls_path.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        for ext in exts:
            # handle both lower/upper extensions
            for f in cls_path.glob(f"*{ext}"):
                shutil.copy2(f, dst_cls)
                copied += 1
            for f in cls_path.glob(f"*{ext.upper()}"):
                shutil.copy2(f, dst_cls)
                copied += 1

    return {
        "mode": "copied_new_filtered",
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "num_classes": len(allowed_class_folders),
        "classes": [p.name for p in allowed_class_folders],
        "num_images": copied,
        "allowed_keywords": allowed_keywords,
    }


def save_classes_json(class_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    print(f"Saved classes JSON: {out_path}")

# ======================================================
# Dataset loading (tf.data) from filtered greenhouse subset
# ======================================================

def build_datasets_from_directory(
    data_dir: Path,
    img_size=(128, 128),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Creates train/val datasets from a folder with class subfolders.
    Returns: train_ds, val_ds, class_names
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    # Performance optimizations
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names





if __name__ == "__main__":
    # Clean start
    K.clear_session()
    gc.collect()

    # Sanity checks
    sanity_checks()

    # Paths
    paths = get_paths()
    print_paths(paths)
    validate_paths(paths)
    print("Paths OK. Ready for energy simulation.")

    # Energy simulation (Day 03 core)
    t_hours, E_profile, E_ideal, energy_at_hour = build_energy_simulator(
        sunrise=7.0,
        sunset=19.0,
        cloud_strength=0.15,
        smooth_k=5,
        seed=42,
    )

    save_energy_plot(
        t_hours,
        E_profile,
        E_ideal,
        THRESHOLD,
        paths["ARTIFACTS_DIR"] / "energy_profile.png",
    )


    # Quick energy checks
    print("Energy check @ 06:00 =", energy_at_hour(6.0))
    print("Energy check @ 12:00 =", energy_at_hour(12.0))
    print("Energy check @ 20:00 =", energy_at_hour(20.0))

    print("Energy simulation ready. Moving to dataset filtering next.")


    # --------------------------------------------------
    # Filter dataset to greenhouse subset
    # --------------------------------------------------
    summary = copy_filtered_dataset(
        src_root=paths["PLANTVILLAGE_DIR"],
        dst_root=paths["FILTERED_DIR"],
        allowed_keywords=ALLOWED_KEYWORDS,
        overwrite=False,   # set True only if you want to rebuild filtered dataset
    )

    print("\n" + "=" * 60)
    print("Greenhouse dataset filtering summary")
    for k, v in summary.items():
        if k == "classes":
            print(f"- {k}: {len(v)} classes")
        else:
            print(f"- {k}: {v}")
    print("=" * 60)

    # save class names for later inference / demo
    # (we save the filtered folder names)
    filtered_class_names = summary.get("classes", [])
    if filtered_class_names:
        save_classes_json(filtered_class_names, paths["CLASSES_JSON"])


    # --------------------------------------------------
    # Load datasets (train/val) from filtered folder
    # --------------------------------------------------
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2

    train_ds, val_ds, class_names = build_datasets_from_directory(
        data_dir=paths["FILTERED_DIR"],
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    print(f"\n Loaded datasets from: {paths['FILTERED_DIR']}")
    print(f"Num classes: {len(class_names)}")
    print("py ./src/energy/day03_solar_window_training.py")
    print("Classes:", class_names)
    # Save classes again (authoritative from TF loader)
    save_classes_json(class_names, paths["CLASSES_JSON"])
