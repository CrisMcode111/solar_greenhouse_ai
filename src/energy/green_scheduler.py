"""
Green scheduler: energy-aware gating for AI workloads.

This module provides a simple decision boundary:
run AI only when solar energy is above a configurable threshold.

The purpose is to enforce an explicit energy constraint, mimicking
solar-powered edge computation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SchedulerConfig:
    threshold: float = 0.5  # energy in [0, 1]


def can_run_ai(energy: float, threshold: float) -> bool:
    """
    Return True if AI computation is allowed under the current energy constraint.
    """
    return energy >= threshold


def schedule_ai(E_solar: np.ndarray, config: SchedulerConfig) -> np.ndarray:
    """
    Vectorized scheduling decision for a full day signal.
    Returns a boolean array of same length as E_solar.
    """
    if not (0.0 <= config.threshold <= 1.0):
        raise ValueError("threshold must be within [0, 1].")
    return E_solar >= config.threshold


def duty_cycle(can_run: np.ndarray) -> float:
    """
    Percentage of time AI is active.
    """
    if can_run.size == 0:
        return 0.0
    return float(can_run.mean() * 100)
