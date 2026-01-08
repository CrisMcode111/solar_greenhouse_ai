from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.system.utils_io import ensure_dir, utc_now_iso


@dataclass
class SystemLogger:
    out_dir: Path
    run_id: str
    csv_path: Path | None = None
    jsonl_path: Path | None = None
    _csv_file: Any | None = None
    _csv_writer: csv.DictWriter | None = None

    def __post_init__(self) -> None:
        ensure_dir(self.out_dir)
        self.csv_path = self.out_dir / "system_log.csv"
        self.jsonl_path = self.out_dir / "events.jsonl"

        # Initialize CSV with header on first write
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        # Header is defined by first row we write (DictWriter needs fieldnames)
        self._csv_writer = None

        # Create/empty JSONL
        self.jsonl_path.write_text("", encoding="utf-8")

    def _ensure_csv_writer(self, row: Dict[str, Any]) -> None:
        if self._csv_writer is not None:
            return
        fieldnames = list(row.keys())
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def log_step(
        self,
        step_idx: int,
        state: Dict[str, Any],
        action: Dict[str, Any],
        constraints: Dict[str, Any],
        outcome: Dict[str, Any],
        why: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = utc_now_iso()

        flat_row: Dict[str, Any] = {
            "run_id": self.run_id,
            "ts_utc": ts,
            "step_idx": step_idx,
            # core
            "why": why,
            # constraints
            "energy_ok": constraints.get("energy_ok"),
            # action
            "action_type": action.get("type"),
            "action_intensity": action.get("intensity"),
            # outcome
            "risk_score": outcome.get("risk_score"),
            "rule_flags": outcome.get("rule_flags"),
        }

        flat_row["threshold_temp_hi"] = 30
        flat_row["threshold_rh_hi"] = 85

        # flatten a few state fields (keep it small; detailed state goes to JSONL)
        for k in ("temp_c", "rh_pct", "light_lux", "energy_available_wh"):
            if k in state:
                flat_row[k] = state[k]

        if extra:
            for k, v in extra.items():
                flat_row[f"extra_{k}"] = v

        self._ensure_csv_writer(flat_row)
        self._csv_writer.writerow(flat_row)
        self._csv_file.flush()

        event = {
            "run_id": self.run_id,
            "ts_utc": ts,
            "step_idx": step_idx,
            "state": state,
            "constraints": constraints,
            "action": action,
            "outcome": outcome,
            "why": why,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
