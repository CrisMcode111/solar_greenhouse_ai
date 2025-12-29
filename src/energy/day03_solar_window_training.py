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

# ======================================================
# Model + Callbacks + Steps per Epoch
# ======================================================

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def build_mobilenetv2_model(num_classes: int, img_size=(128, 128), lr: float = 1e-3):
    """
    MobileNetV2 backbone (frozen) + small classification head.
    Good for quick Day03 training on CPU.
    """
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

    # MobileNetV2 expects inputs in a specific range; this layer does it
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False  # Day03: keep it frozen (fast & stable)

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(artifacts_dir: Path):
    ckpt_path = artifacts_dir / "best_model.keras"
    log_path = artifacts_dir / "training_log.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(str(log_path), append=True),
    ]
    return callbacks, ckpt_path, log_path


def compute_steps(ds):
    """
    Robust steps computation for tf.data datasets.
    Returns int or None.
    """
    try:
        card = tf.data.Dataset.cardinality(ds).numpy()
        if card == tf.data.INFINITE_CARDINALITY or card == tf.data.UNKNOWN_CARDINALITY:
            return None
        return int(card)
    except Exception:
        return None

# ======================================================
# Solar Training Loop (Frugal AI)
# ======================================================

def solar_training_loop(
    model,
    train_ds,
    val_ds,
    energy_at_hour,
    threshold: float,
    callbacks,
    model_path: Path,
    hist_csv: Path,
    num_solar_days: int = 3,
    hours_to_sim=range(6, 21),   # 06..20
    train_steps_per_hour: int = 80,   # IMPORTANT: mini-epoch (fast). You can raise later.
    val_steps: int = 50,
):
    """
    Trains 1 mini-epoch when energy >= threshold.
    Otherwise sleeps (optionally evaluates).
    Saves model + history CSV.
    """
    history_log = []

    for day in range(1, num_solar_days + 1):
        print(f"\n====================== Solar Day {day} ======================")

        for h in hours_to_sim:
            Eh = float(energy_at_hour(float(h)))

            if Eh >= threshold:
                print(f"[Day {day} | {h:02d}:00] ☀️  E={Eh:.2f} ≥ {threshold} → TRAIN (mini-epoch)")

                hist = model.fit(
                    train_ds,
                    epochs=1,
                    steps_per_epoch=train_steps_per_hour,
                    validation_data=val_ds,
                    validation_steps=val_steps,
                    verbose=1,
                    callbacks=callbacks,
                )

                val_acc = float(hist.history.get("val_accuracy", [None])[-1])
                val_loss = float(hist.history.get("val_loss", [None])[-1])
                trained = True

            else:
                print(f"[Day {day} | {h:02d}:00] 🌙  E={Eh:.2f} < {threshold} → SLEEP (no training)")

                # light eval only (optional)
                try:
                    val_loss, val_acc = model.evaluate(val_ds, steps=val_steps, verbose=0)
                    val_loss = float(val_loss)
                    val_acc = float(val_acc)
                except Exception:
                    val_loss, val_acc = None, None

                trained = False

            history_log.append(
                {
                    "day": int(day),
                    "hour": int(h),
                    "E": Eh,
                    "trained": trained,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )

    # Save
    df = pd.DataFrame(history_log)
    hist_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(hist_csv, index=False)
    print(f"\n✅ Saved solar history CSV: {hist_csv} ({len(df)} rows)")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"✅ Saved final model: {model_path}")

    return df

# ======================================================
# Plots: Energy vs Validation metrics
# ======================================================

def plot_energy_vs_metrics(hist_csv: Path, out_dir: Path):
    """
    Generate plots:
    - Energy vs val_accuracy
    - Energy vs val_loss
    """
    if not hist_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {hist_csv}")

    df = pd.read_csv(hist_csv)

    # Keep only rows where we have metrics
    df = df.dropna(subset=["val_acc", "val_loss"])

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Energy vs Accuracy ---
    plt.figure(figsize=(6, 4))
    plt.scatter(df["E"], df["val_acc"], alpha=0.7)
    plt.xlabel("Solar Energy (E)")
    plt.ylabel("Validation Accuracy")
    plt.title("Energy vs Validation Accuracy")
    plt.grid(True)
    acc_path = out_dir / "energy_vs_val_accuracy.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    # --- Energy vs Loss ---
    plt.figure(figsize=(6, 4))
    plt.scatter(df["E"], df["val_loss"], alpha=0.7)
    plt.xlabel("Solar Energy (E)")
    plt.ylabel("Validation Loss")
    plt.title("Energy vs Validation Loss")
    plt.grid(True)
    loss_path = out_dir / "energy_vs_val_loss.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(" Saved plots:")
    print(f" - {acc_path}")
    print(f" - {loss_path}")





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

        # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    num_classes = len(class_names)
    model = build_mobilenetv2_model(
        num_classes=num_classes,
        img_size=IMG_SIZE,
        lr=1e-3,
    )
    model.summary()

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    callbacks, ckpt_path, log_path = build_callbacks(paths["ARTIFACTS_DIR"])
    print(f" Checkpoint: {ckpt_path}")
    print(f" CSV log   : {log_path}")

    # --------------------------------------------------
    # Steps per epoch
    # --------------------------------------------------
    STEPS_PER_EPOCH = compute_steps(train_ds)
    VALIDATION_STEPS = compute_steps(val_ds)

    print(f"STEPS_PER_EPOCH  = {STEPS_PER_EPOCH}")
    print(f"VALIDATION_STEPS = {VALIDATION_STEPS}")

    # Safety fallback (in case cardinality is None)
    if STEPS_PER_EPOCH is None:
        STEPS_PER_EPOCH = 50
        print(" STEPS_PER_EPOCH unknown → fallback to 50")
    if VALIDATION_STEPS is None:
        VALIDATION_STEPS = 10
        print(" VALIDATION_STEPS unknown → fallback to 10")


    # --------------------------------------------------
    # Solar training loop (fast settings for CPU)
    # --------------------------------------------------
    NUM_SOLAR_DAYS = 3
    HOURS_TO_SIM = range(6, 21)   # 06..20

    # IMPORTANT: keep small first, then increase if you want
    TRAIN_STEPS_PER_HOUR = 80
    VAL_STEPS = 50

    hist_df = solar_training_loop(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        energy_at_hour=energy_at_hour,
        threshold=THRESHOLD,
        callbacks=callbacks,
        model_path=paths["MODEL_PATH"],
        hist_csv=paths["HIST_CSV"],
        num_solar_days=NUM_SOLAR_DAYS,
        hours_to_sim=HOURS_TO_SIM,
        train_steps_per_hour=TRAIN_STEPS_PER_HOUR,
        val_steps=VAL_STEPS,
    )

    print("\n Done. Next: plot E vs val_acc/val_loss.")

        # --------------------------------------------------
    # Final plots (Energy vs metrics)
    # --------------------------------------------------
    plot_energy_vs_metrics(
        hist_csv=paths["HIST_CSV"],
        out_dir=paths["ARTIFACTS_DIR"],
    )

    print(" Plots generated. Day03 complete.")

