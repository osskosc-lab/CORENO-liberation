# Preregistered protocol: CORENO Multi-Agent Phase 1 Falsification v1.0

## Scope

The target is a limited controller claim, not a claim about metaphysics or an
ontology:

\[
L^{shift}_{CORENO} < L^{shift}_{StrongBaseline}.
\]

The experiment asks whether a controller that keeps multiple hypotheses alive,
records dissent, can abstain at a cost, and partially rebuilds its integration
weights adapts better immediately after a correlation reversal.

## Frozen environment

Each episode has 3,000 labelled binary decisions. The feature vector has:

| Feature | Stable phase | Unknown-shift phase |
|---|---:|---:|
| causal `z` | 70% label agreement | 70% label agreement |
| spurious `s` | 90% label agreement | 10% label agreement |
| context `c` | 58% label agreement, boundary cue | same |
| noise `n` | random | random |

The stable phase is steps 0-999. Shifts begin at 1000 and 2500. The primary
window is steps **1000-1299** only; full-episode outcomes are descriptive.

## Fixed agents and controls

Five online logistic agents use identical data, label feedback, learning-rate,
and update count in every condition. Fixed feature masks make their hypotheses
diverse. The primary comparator is `strong`: confidence weighting, online Hedge,
error-based change detection, and costed abstention.

The confirmatory comparison is `coreno_full / strong`. `single`, `majority`,
and all ablations are secondary controls.

## Loss and safety

Correct action has loss 0, wrong action loss 1, and self-stop loss 0.25.
The treatment safety gates are:

- high-confidence-wrong-rate difference from `strong` <= 0.01;
- treatment STOP rate <= 0.35.

## Main statistic and stop rule

For each of 100 fixed seeds, calculate its average primary-window loss. Use a
seed-paired nonparametric bootstrap with 10,000 replicates:

\[
Q = \frac{\overline{L}_{coreno\_full}}{\overline{L}_{strong}}.
\]

- **SUPPORTED:** upper 95% CI of `Q` <= 0.90 and both safety gates pass.
- **FALSIFIED:** lower 95% CI of `Q` >= 1.00.
- **INCONCLUSIVE:** every other outcome.

There is no optional continuation rule. Any additional seeds, retuning, new
environment, or amended control is a separately registered phase.

## Mechanism checks

The secondary CORENO variants remove one component: plural-weight floor,
dissent flag, self-stop, or reconstruction. Two nulls corrupt the relationship
between agent performance history and agent identity. These checks may explain
a result but cannot upgrade an inconclusive primary outcome to support.
