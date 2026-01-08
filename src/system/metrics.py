from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.system.utils_io import write_json


def compute_metrics_from_csv(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    metrics: Dict[str, Any] = {
        "rows": int(len(df)),
        "run_ids": sorted(df["run_id"].unique().tolist()) if "run_id" in df.columns else [],
    }

    if "energy_ok" in df.columns:
        eo = df["energy_ok"]
        metrics["energy_ok_rate"] = float(pd.to_numeric(eo, errors="coerce").fillna(0).mean())

    if "action_type" in df.columns:
        metrics["action_counts"] = df["action_type"].value_counts(dropna=False).to_dict()

    if "risk_score" in df.columns:
        rs = pd.to_numeric(df["risk_score"], errors="coerce")
        metrics["risk_score_mean"] = float(rs.mean())
        metrics["risk_score_p95"] = float(rs.quantile(0.95))

    metrics["has_why"] = bool(("why" in df.columns) and df["why"].notna().all())
    metrics["has_actions"] = bool(("action_type" in df.columns) and df["action_type"].notna().any())

    # Auditability rates
    if "why" in df.columns:
        metrics["why_nonempty_rate"] = float(df["why"].astype(str).str.strip().ne("").mean())
    else:
        metrics["why_nonempty_rate"] = 0.0

    if "action_type" in df.columns:
        metrics["action_nonempty_rate"] = float(df["action_type"].astype(str).str.strip().ne("").mean())
    else:
        metrics["action_nonempty_rate"] = 0.0

    return metrics



def write_metrics_summary(out_dir: Path) -> Path:
    csv_path = out_dir / "system_log.csv"
    summary = compute_metrics_from_csv(csv_path)

    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        summary["schema_mapping_coverage"] = compute_schema_coverage_from_manifest(manifest_path)

    out_path = out_dir / "metrics_summary.json"
    write_json(out_path, summary)
    return out_path


def compute_schema_coverage_from_manifest(manifest_path: Path) -> float:
    """
    Returns fraction of expected system columns that were successfully mapped.
    """
    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    colmap = manifest.get("colmap", {})
    if not colmap:
        return 0.0

    total = len(colmap)
    matched = sum(1 for v in colmap.values() if v is not None)

    return matched / total if total > 0 else 0.0

