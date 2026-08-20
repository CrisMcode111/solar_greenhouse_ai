# Solar Greenhouse AI

## Overview

**Solar Greenhouse AI** is an experimental applied-AI project exploring how greenhouse decision systems can operate under environmental, biological, and energy constraints.

The project investigates a central question:

> **How should a greenhouse decision system prioritize actions when plant needs, environmental conditions, and available energy compete?**

Rather than treating greenhouse intelligence as a purely predictive problem, the project explores a lightweight decision architecture in which environmental perception, plant-stress assessment, operational rules, and energy availability contribute to the final decision.

The project originated from an interest in **solar-powered and energy-efficient greenhouse systems** and gradually evolved from early frugal-AI experiments toward a broader investigation of **resource-aware decision-making**.

The current system is a **research prototype**. It has not been deployed as a production greenhouse controller and has not yet been validated on a commercial greenhouse.

---

## Research Motivation

Greenhouse operation is a constrained decision problem.

A control or decision-support system may need to consider simultaneously:

- temperature and humidity,
- light conditions,
- plant stress,
- soil or substrate conditions,
- operational priorities,
- available energy,
- and the cost or feasibility of an intervention.

These variables can create competing objectives.

An intervention may be beneficial for the crop but expensive in terms of energy. Another action may be delayed safely. Under limited or variable energy availability — including solar-powered scenarios — not every possible intervention can necessarily be treated as equally urgent.

This project explores how these constraints can be represented explicitly in the decision process.

The objective is not to build a fully autonomous greenhouse at this stage, but to investigate the architecture required for **resource-aware, explainable operational decisions**.

---

## From Frugal AI to Decision Intelligence

The project initially focused on **frugal AI**:

- lightweight machine-learning models,
- model simplification,
- pruning,
- inference efficiency,
- and compatibility with low-resource computing environments.

This remains relevant.

However, experimentation led to a broader question.

Reducing the computational cost of a model is useful, but computational efficiency alone does not determine whether an AI system makes an appropriate operational decision.

The project therefore evolved from:

```text
How can the model use fewer computational resources?
```

toward:

```text
How should the system make decisions when operational resources themselves are constrained?
```

This distinction became central to the current research direction.

---

## System Architecture

The project explores a layered decision architecture rather than relying on a single predictive model.

Conceptually:

```text
Environmental / Sensor State
            │
            ▼
       Perception
            │
            ▼
   Plant Stress Assessment
            │
            ▼
      Rules / Policy
            │
            ├──────────────┐
            │              │
            ▼              ▼
     Candidate Action   Energy /
                        Resource State
            │              │
            └──────┬───────┘
                   ▼
               Arbitrator
                   │
                   ▼
          Operational Decision
```

The architecture separates several reasoning responsibilities.

### Perception

Environmental information is transformed into a state that can be used by downstream decision logic.

### Plant-Stress Assessment

Environmental conditions are interpreted in relation to potential plant stress rather than treated only as isolated sensor values.

### Rules / Policy

Operational logic determines which actions may be appropriate under the current conditions.

### Energy / Resource State

The system considers whether sufficient resources are available for a proposed action.

### Arbitration

When operational priorities and resource constraints compete, the arbitration layer determines which action should take precedence.

This layered structure makes it possible to inspect **why** a decision was produced rather than treating the model as an opaque end-to-end controller.

---

## Energy-Aware Decision Logic

Energy is treated as more than a background efficiency metric.

It can become an explicit constraint on operational decisions.

Conceptually, the system investigates situations such as:

```text
Plant stress detected
        +
Operational action available
        +
Energy availability constrained
        ↓
Which action should be prioritized?
```

This is particularly relevant to systems where energy availability may vary over time.

Solar generation is one example, but the architecture can conceptually accommodate other energy contexts such as:

- limited grid availability,
- battery constraints,
- dynamic electricity pricing,
- or mixed energy sources.

The repository does **not** currently implement a complete photovoltaic installation or physical energy-management system.

Solar energy represents the original resource constraint that motivated the project.

---

## Current Implementation

The repository contains the experimental components developed during the different phases of the project, including work related to:

- environmental and system-state representation,
- plant-stress-oriented reasoning,
- operational rules and policies,
- energy-aware decision logic,
- arbitration between competing conditions,
- lightweight machine-learning experimentation,
- frugal-AI and model-efficiency exploration,
- and experimental notebooks used to investigate system behaviour.

The emphasis is currently on **decision architecture and experimentation**, rather than production deployment.

---

## Experimental Approach

The project follows an iterative research workflow:

```text
Problem
   ↓
Hypothesis
   ↓
Architecture
   ↓
Implementation
   ↓
Experiment
   ↓
Observation
   ↓
Refined hypothesis
```

The purpose of the experiments is not simply to maximize model accuracy.

Instead, they investigate questions such as:

- Does the system react differently when energy availability changes?
- Can plant-related priorities remain visible when resources are constrained?
- Can operational rules and resource constraints be evaluated separately?
- Can conflicting signals be resolved explicitly?
- Can the reasoning behind a decision remain inspectable?

This approach treats the repository as an evolving **applied research system**, rather than a finished product.

---

## Phase 1 — Energy-Aware Training Logic

Phase 1 focused on introducing and testing energy-aware logic within the experimental system.

The objective was to investigate how resource constraints could influence system behaviour while preserving a lightweight computational approach.

This phase is complete and documented in the repository.

Its main contribution was to move the project beyond model-efficiency experimentation toward **energy-aware system behaviour**.

---

## Current Research Direction

Subsequent analysis of the greenhouse technology landscape showed that greenhouse automation, climate control, energy optimization, and data-driven growing are already mature industrial fields.

This changed the framing of the project.

The most useful next question is therefore not:

> **Can AI automate a greenhouse?**

but rather:

> **Can a lightweight decision layer identify resource-efficient operational choices while keeping the system within predefined crop-safe conditions?**

This leads toward a narrower experimental direction:

### Retrofit Decision Intelligence

Instead of replacing existing greenhouse infrastructure, a future version could operate as a decision-support layer over existing data and equipment.

The system could:

```text
observe
   ↓
assess
   ↓
recommend
   ↓
explain
   ↓
measure
```

without directly controlling greenhouse actuators.

This **shadow-mode** approach would allow recommendations to be evaluated before autonomous control is considered.

---

## Next Validation

The next meaningful validation would involve running the decision architecture against **real or historical greenhouse data**.

The objective would be to test whether the system can identify operational decisions that reduce unnecessary resource consumption while maintaining predefined crop-safe conditions.

A future experiment could evaluate three types of metrics:

### Resource impact

```text
estimated energy / resource saving
```

### Agronomic guardrail

```text
time maintained within predefined crop-safe conditions
```

### Operational impact

```text
interventions avoided
or
operator attention saved
```

The purpose would be to move from architectural feasibility toward measurable operational value.

---

## Current Limitations

This repository should be interpreted as an experimental research prototype.

Current limitations include:

- no deployment in a commercial greenhouse,
- no autonomous control of physical actuators,
- no proprietary greenhouse hardware,
- no complete photovoltaic or battery system,
- no agronomic field validation,
- no demonstrated commercial energy savings,
- and reliance on experimental or simulated conditions in parts of the current work.

These limitations define the boundary between what has been explored technically and what still requires real-world validation.

---

## What This Project Is — and Is Not

### It is

- an applied-AI research prototype,
- an exploration of constrained decision systems,
- an investigation of energy-aware greenhouse intelligence,
- an experiment in lightweight and resource-aware AI,
- and a framework for studying interactions between environmental, biological, and operational constraints.

### It is not

- a production greenhouse controller,
- a validated agronomic system,
- a commercial greenhouse automation platform,
- a replacement for existing industrial climate computers,
- or a claim of autonomous greenhouse operation.

---

## Project Evolution

The project has evolved through several stages:

```text
Frugal AI
    ↓
Model efficiency
    ↓
Energy-aware logic
    ↓
Plant / environment reasoning
    ↓
Constraint arbitration
    ↓
Decision intelligence
    ↓
Real-world validation (next)
```

This evolution reflects one of the central findings of the project:

> **In real-world AI systems, efficiency is not only about reducing the computational cost of the model. It is also about making better decisions under physical, biological, and resource constraints.**

---

## Project Origin

Solar Greenhouse AI originated during a **Generative AI & Machine Learning bootcamp** as an applied exploration of frugal AI and sustainable computing.

It was subsequently continued as an independent project, with the research scope expanding toward energy-aware decision systems and applied AI for physical and biological environments.

---

## Status

**Status:** Experimental research prototype  
**Current stage:** Architecture and decision-logic exploration  
**Completed:** Phase 1 — Energy-Aware Training Logic  
**Next validation:** Shadow-mode evaluation using real or historical greenhouse data

Development is intentionally incremental. New functionality is added when it supports a specific research question or validation objective.

---

## Broader Research Interest

This project is part of a broader interest in **Applied AI for food and agricultural systems**, particularly systems in which AI must operate within real-world constraints rather than purely informational environments.

Areas of interest include:

- resource-aware AI,
- decision-support systems,
- sustainable and frugal computing,
- food and agricultural systems,
- physical and biological constraints,
- explainable operational decisions,
- and AI-assisted product and system innovation.

---

## Disclaimer

Solar Greenhouse AI is an independent experimental project.

The repository is intended for research, learning, and technical exploration and should not be used to control agricultural equipment or make production-critical agronomic decisions without appropriate validation and domain expertise.
