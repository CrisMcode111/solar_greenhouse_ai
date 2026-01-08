from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import pandas as pd


@dataclass(frozen=True)
class SchemaSpec:
    # fiecare grup = “trebuie să existe MINIM una din coloanele astea”
    required_any: List[List[str]]
    # numeric columns (dacă există) – le verificăm că sunt “cam numerice”
    numeric_any: List[List[str]]


DEFAULT_SCHEMA = SchemaSpec(
    required_any=[
        # time column (acceptă mai multe denumiri)
        ["ts", "timestamp", "datetime", "time"],

        # temperature (inside or generic)
        ["inside_temp_c", "temp_c", "t_inside_c", "temperature_c"],

        # humidity
        ["inside_rh_pct", "rh_pct", "humidity_pct"],

        # energy coupling key
        ["energy_ok", "energy_available_wh", "energy_wh"],
    ],
    numeric_any=[
        ["inside_temp_c", "temp_c", "t_inside_c", "temperature_c"],
        ["inside_rh_pct", "rh_pct", "humidity_pct"],
        ["energy_available_wh", "energy_wh"],
        ["light_lux", "inside_light_lux"],
    ],
)


def _first_present(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def check_schema(df: pd.DataFrame, spec: SchemaSpec = DEFAULT_SCHEMA) -> Tuple[bool, str, Dict[str, Any]]:
    ok = True
    notes: List[str] = []
    chosen: Dict[str, str] = {}

    # required groups
    for group in spec.required_any:
        hit = _first_present(df, group)
        if hit is None:
            ok = False
            notes.append(f"Missing one of: {group}")
        else:
            chosen[" OR ".join(group)] = hit

    # numeric sanity (only for those present)
    coercion_issues = []
    for group in spec.numeric_any:
        hit = _first_present(df, group)
        if hit is None:
            continue
        coerced = pd.to_numeric(df[hit], errors="coerce")
        frac_nan = float(coerced.isna().mean())
        if frac_nan > 0.20:
            ok = False
            coercion_issues.append((hit, frac_nan))

    if coercion_issues:
        notes.append(f"Numeric coercion issues (col, frac_nan): {coercion_issues}")

    summary = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "matched_columns": chosen,
    }

    report_txt = "SCHEMA CHECK\n" + ("\n".join(notes) if notes else "OK")
    return ok, report_txt, summary

