# Phase 3 — System View & Controlled Autonomy

## Overview

Phase 3 transforms the project from a collection of modules into a **coherent, explainable system**.  
The greenhouse controller now operates as an integrated decision-making pipeline, capable of:

- perceiving the environment,
- reasoning in discrete states,
- comparing multiple decision strategies,
- arbitrating safely between them,
- and explaining every final action.

This phase focuses on **system-level behavior**, not on model performance.

---

## Objectives of Phase 3

Phase 3 was designed to answer the following questions:

- *How does the system decide what to do at each timestep?*
- *How can decisions be explained and audited?*
- *How do rules and policies coexist without conflict?*
- *How is plant stress managed in a seasonal, energy-aware context?*

The result is a **controlled autonomy loop**, not a black box.

---

## Architecture Summary

Phase 3 builds on the previous phases:

- **Phase 1 — Energy Logic**  
  Provides authority constraints (energy availability, conservation rules).

- **Phase 2 — Perception & Risk Detection**  
  Produces normalized sensor data and plant risk flags.

Phase 3 integrates both into a runtime system:
Sensors → Perception → Discrete State → Rules & Policy → Arbitration → Final Action


---

## Day-by-Day Structure

### Day 10 — System Skeleton & Pipeline Entry Point
- Introduced a single entry point to run the full pipeline.
- Unified dataset loading, schema validation, and reporting.
- Produced a reproducible system report and run manifest.

**Outcome:**  
A runnable system, not just isolated scripts.

---

### Day 11 — Observability & Logging
- Added system-level logging:
  - state features
  - actions
  - constraints
  - outcomes
  - explanations (`why`)
- Introduced auditability metrics.

**Outcome:**  
Every decision leaves a trace answering:  
> *“What happened, and why?”*

---

### Day 12 — Decision Loop v1 (Rules Engine)
- Implemented a minimal rule-based controller.
- Decisions based on thresholds and plant risk flags.
- Added a confusion table to detect:
  - energy conflicts
  - missed risks
  - rule gaps

**Outcome:**  
A deterministic, explainable baseline controller.

---

### Day 13 — Discrete State Mapping & Explanations
- Introduced a shared **state vocabulary**:
  - Temperature: `LOW | OK | HIGH`
  - Soil: `DRY | OK | WET`
  - Humidity: `LOW | OK | HIGH`
  - Energy: `OK | NOT_OK`
  - Outside context: `FREEZING | COLD | MILD | HOT`
- Each timestep now has:
  - a `state_code`
  - a human-readable explanation
- Produced state timelines and distributions.

**Outcome:**  
The system reasons in **states**, not raw numbers.

---

### Day 14 — Rules vs Policy Comparison
- Introduced a second decision strategy: **policy-based control**.
- Compared rule-based and policy-based actions per state.
- Quantified agreement and disagreement by state.

**Outcome:**  
Divergences are *measured and explained*, not treated as bugs.

---

### Day 15 — Stress Index & Arbitration (Controlled Autonomy)
- Introduced a **plant stress index**:
  - heat stress
  - water stress
  - cold vent risk
- Implemented an **arbitration layer**:
  - hard safety gates (energy, freezing)
  - soft overrides based on stress severity
  - rescue logic when rules are idle
- Produced final actions with explicit decision sources:
  - `rules`
  - `policy_override`
  - `gated_energy`
  - `gated_freezing`

**Outcome:**  
A single, final, safe, explainable decision per timestep.

---

## Key Concepts Introduced

### Discrete State Language
The system now operates using a shared symbolic language:

T_HIGH | S_DRY | H_OK | E_OK | O_HOT


This enables:
- reasoning
- comparison
- explainability
- future learning

---

### Rules vs Policy
- **Rules**: deterministic, conservative, threshold-based.
- **Policy**: contextual, season-aware, plant-oriented.

Phase 3 does not choose one over the other.  
It **orchestrates** them.

---

### Arbitration & Safety
Autonomy is never absolute.

Hard constraints always apply:
- no energy → no actuation
- freezing outside → no ventilation

Soft constraints allow adaptation based on stress and context.

---

## Artifacts Produced

Phase 3 produces structured, inspectable artifacts:

- system logs and metrics (Day 11)
- action traces and confusion tables (Day 12)
- state timelines and distributions (Day 13)
- rules vs policy comparisons (Day 14)
- final actions, arbitration logs, and stress summaries (Day 15)

These artifacts enable:
- debugging
- tuning
- backtesting
- communication with non-technical stakeholders

---

## Phase 3 Status

**Phase 3 is complete.**

It delivers:
- a full system loop,
- controlled autonomy,
- explainable decisions,
- and a solid foundation for learning-based control or optimization.

---

## Next Steps (Optional)

- **Phase 4**: Learning & Adaptation  
  (policy learning, adaptive thresholds, reinforcement signals)

- **Day 16 (optional)**: Backtesting & tuning  
  (stress calibration, seasonal profiles, performance plots)

---

**Phase 3 establishes the system’s “thinking layer”.**  
Everything that follows builds on this structure.

