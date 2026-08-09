# CORENO Multi-Agent Phase 1 Falsification

This repository contains a reproducible, deliberately narrow test of an engineering claim:

> With identical agents, observations, and online learning budget, does a CORENO-style
> integration controller lower decision loss immediately after an unseen environment shift
> relative to a strong multi-agent baseline?

It does **not** test or establish CORENO ontology, consciousness, liberation, or any
metaphysical conclusion.

## Preregistered decision rule

The primary metric is the seed-paired ratio

`Q = mean(post-shift loss, CORENO full) / mean(post-shift loss, strong baseline)`

where post-shift loss is the average abstention-aware loss in steps 1000-1299. The result is:

- **SUPPORTED** only if the bootstrap 95% upper confidence interval of `Q` is at most 0.90,
  the high-confidence-wrong-rate difference is at most 0.01, and the STOP rate is at most 0.35.
- **FALSIFIED** if the bootstrap 95% lower confidence interval of `Q` is at least 1.00.
- **INCONCLUSIVE** otherwise.

All 100 seeds are fixed before interpretation. The script does not add seeds after inspecting
the result.

## Conditions

`single`, `majority`, and `strong` are reference controls. `coreno_full` is compared against
`strong` for the confirmatory result. The remaining CORENO conditions are explanatory
ablations/null controls only.

The strong baseline uses confidence weighting, online Hedge, error-based change detection,
and costed abstention. CORENO uses the same agents plus bounded plural weights, an explicit
dissent flag, costed self-stop, and partial weight reconstruction. All conditions receive the
same per-seed generated environment and label feedback.

## Run

```bash
python3 coreno_multiagent_phase1.py --seeds 100 --bootstrap 10000 --outdir results
python3 generate_report.py --results results --output results/CORENO_MultiAgent_Phase1_Report.pdf
```

The first command writes machine-readable CSV files and figures. The second produces a
visual report that states the frozen protocol, numerical outcome, safety-gate status, and
the appropriately limited conclusion.

## Environment

During training-like stable exposure, the spurious feature agrees with the label 90% of the
time. At each unseen shift it agrees only 10% of the time, while a causal feature remains
70% accurate. The five fixed online logistic agents differ only in feature masks, creating
diversity without giving the treatment access to privileged information.

Dependencies: Python 3.10+, NumPy, pandas, matplotlib, and reportlab.
