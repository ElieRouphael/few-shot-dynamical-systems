# Learning to identify new dynamical systems from a few experiments

A hands-on introduction to **pretraining, few-shot adaptation, and experiment design for families
of dynamical systems**: the core ideas behind foundation models for dynamics, in a setting small
enough that every step can be inspected.

> Engineered systems come in families. Can knowledge extracted from many previously identified
> systems identify a new member of the family from a few seconds of data, and which experiment
> should we run to learn it fastest?

## Start here

| | |
|---|---|
| **Tutorial notebook** | [`notebooks/few_shot_system_identification_tutorial.ipynb`](notebooks/few_shot_system_identification_tutorial.ipynb) (executed, with all figures) |
| **Technical report** | [`report/tutorial_report.pdf`](report/tutorial_report.pdf): methods, derivations, evaluation protocol, results and limitations |

The notebook is self-contained (NumPy, SciPy, Matplotlib) and runs in about a minute.

## What the tutorial covers

| § | Question | Method |
|---|---|---|
| 1–2 | What is the system, and what can we measure? | family of nonlinear oscillators, position-only measurements, coloured disturbance, unmodelled forces |
| 3 | How well can one system be identified from scratch? | output-error least squares, identifiability |
| 4 | What do many systems have in common? | per-system identification + identifiability-weighted (noise-whitened) PCA |
| 5 | Can shared knowledge identify a new system from little data? | MAP estimation of a 2-D context, Laplace uncertainty |
| 6 | When is pretraining worth it? | data-budget study on fresh systems |
| 7 | Which experiment should we run? | Bayesian D-optimal experiment design |
| 8 | When does the shared knowledge not apply? | misfit test for out-of-family systems |
| 9 | Does a better model give better control? | model-based feedforward tracking |

The model family is

$$
\dot q=v,\qquad \dot v=-kq-dv+b\,\varphi(u_{t-1})-\alpha q^3\;\underbrace{-\;\gamma q^5+r+w_t}_{\text{unknown to the learner}},\qquad y_t=q_t+\nu_t ,
$$

with coefficients $[k,d,b,\alpha]$ generated from two hidden design variables plus individual deviations.
The learner never sees the coefficients, the velocity, the disturbance or the family structure.

## Evaluation protocol

Every claim is evaluated on **freshly drawn systems** that were not used for pretraining, with paired
comparisons, random baselines that are never scored by the design criterion, and bootstrap intervals.
Models are scored on their prediction of the system's disturbance-free response to a new input.

## Reproduce

```bash
python -m venv .venv
python -m pip install -r requirements.txt
jupyter lab                                   # open the tutorial notebook
```

Rebuild the notebook and the report from source (requires `pdflatex`):

```bash
python tools/build_report.py
```

`tools/build_tutorial_notebook.py` defines the notebook; `tools/build_report.py` executes it, exports
the figures and metrics to `report/`, and compiles `report/tutorial_report.tex`. All numbers in the report
are generated from the executed notebook.

## Project structure

```text
notebooks/
  few_shot_system_identification_tutorial.ipynb   # the tutorial (executed)
report/
  tutorial_report.pdf / .tex                      # technical report
  results.tex, tutorial_metrics.json, figures/    # generated from the notebook
tools/
  build_tutorial_notebook.py                      # notebook source
  build_report.py                                 # execute notebook + build report
```

## Scope

This is a transparent, small-scale instance of the pipeline, not a foundation model. The physics-based
backbone and PCA would be replaced at scale by learned structured state-space models and context
encoders; the questions it raises (identifiability of learned contexts, calibrated uncertainty under
coloured noise, safe excitation, detection of out-of-family systems) are the research agenda.

## License

MIT License, Copyright © 2026 Elie Rouphael.
