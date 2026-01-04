# Phase 2 – Perception Logic

## Purpose
Phase 2 defines **how the system observes the greenhouse**.

Its role is to transform raw environmental dynamics into a **structured,
realistic, and learnable perception dataset**, while remaining independent
from decision-making and control.

Phase 2 answers the question:
> *What does the system see, and how reliable is that perception?*

---

## Scope
Phase 2 spans **Days 06 to 09** and includes:

- synthetic sensor generation
- realistic sensor noise and corruptions
- coupling with frozen energy constraints
- a first energy-aware learning baseline

No control, scheduling, or optimization logic is included here.

---

## Architectural Position

Environment
↓
[ Phase 2 – Perception ]
↓
[ Phase 3 – System / Control ]


Phase 2 is a **library layer**, not a runtime controller.

---

## Core Components

### Synthetic Sensors (Day 06)
Implemented in:
src/perception/sensors.py


Provides:
- coherent time-series signals (hourly)
- indoor and outdoor environment modeling
- soil, light, and actuator states

Guarantees:
- temporal consistency
- physical plausibility
- reproducibility (seeded generation)

---

### Sensor Noise & Corruptions (Day 07)
Implemented in:
src/perception/noise.py


Introduces:
- random missing values
- missing blocks (sensor outages)
- spikes / outliers
- slow sensor drift

Purpose:
- approximate real-world sensor imperfections
- prevent overly optimistic learning

---

### Energy Coupling (Day 08)
Implemented via:
src/perception/energy_bridge.py


Principle:
- energy logic is **authoritative and frozen**
- perception never decides energy availability
- `energy_ok` is overridden using Phase 1 logic

This establishes a strict separation:
- perception observes
- energy authorizes

---

### First ML Baseline (Day 09)
Implemented in:
src/perception/day09_first_ml.py


Features:
- supervised learning task: predict `inside_temp_c(t+1)`
- linear regression baseline
- training gated by `energy_ok`

Purpose:
- validate learnability
- verify pipeline integrity
- establish a reference point (not performance optimization)

---

## Artifacts Produced

Phase 2 produces the following versioned artifacts:

- Day 06: clean synthetic sensor dataset
- Day 07: noisy sensor dataset
- Day 08: energy-coupled perception dataset
- Day 09: ML baseline metrics, predictions, and plots

Artifacts are stored under:
artifacts/day06/
artifacts/day07/
artifacts/day08/
artifacts/day09/


---

## Guarantees of Phase 2
At the end of Phase 2, the system guarantees:

- realistic, imperfect sensor data
- explicit energy constraints applied to perception
- a validated, end-to-end learnable dataset
- clear separation between observation and decision

Phase 2 is now considered **stable**.

---

## Out of Scope
Phase 2 explicitly does NOT include:
- control logic
- scheduling
- optimization
- feedback loops
- policy learning

These belong to Phase 3.

---

## Transition to Phase 3
Phase 3 (starting Day 10) treats Phase 2 as a **black-box perception module**
and focuses on:

- system-level orchestration
- decision-making under constraints
- control and scheduling strategies

Phase 2 is no longer modified, only consumed.
