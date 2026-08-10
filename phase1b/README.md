# CORENO Multi-Agent Phase 1B

## Mechanism-Isolation Confirmatory Falsification

Phase 1 ended **INCONCLUSIVE** under its frozen 10% improvement rule. Phase 1B does not add seeds to that result. It is a new preregistered experiment designed to isolate the two mechanisms that survived the Phase 1 ablation most clearly: plurality and persistent agent identity.

### Primary question

Does the adaptation advantage come from preserving heterogeneous agent plurality and persistent agent identity, rather than from arbitrary agent labels, generic ensembling, extra model updates, or controller overhead?

### Critical controls

- `static_id_permute`: relabel/reorder agents from t=0; must be equivalent to `coreno_full`.
- `dynamic_id_shuffle`: destroy persistent identity after the shift.
- `profile_shuffle`: swap behavioral profiles at the shift while retaining identity slots.
- `cloned_agents`: remove behavioral heterogeneity while keeping five agent slots.
- `collapsed_plurality`: retain all predictions but collapse them before identity-sensitive weighting.

### Environment panel

Five frozen synthetic environments are used:

1. correlation reversal (Phase 1 replication environment)
2. minority-correct regime (dissent stress test)
3. high-confidence trap (self-stop stress test)
4. repeated regime changes (reconstruction stress test)
5. no-shift null (unnecessary-intervention cost)

### Confirmatory scale

- 200 frozen seeds
- 5 environments
- 16 conditions
- primary post-shift window: steps 1000-1299
- 10,000 cluster-bootstrap resamples

The 32-seed smoke test is engineering-only and cannot be cited as scientific evidence.

### Frozen decision rules

Primary global ratio:

`Q_global = loss(coreno_full) / loss(strong)`

- strong support: UCI95 <= 0.90 plus safety gates
- replicated superiority: UCI95 < 1.00 plus safety gates
- falsified: LCI95 >= 1.00
- otherwise inconclusive

Primary identity mechanism ratio:

`Q_id = loss(coreno_full) / loss(dynamic_id_shuffle)`

- mechanism support: UCI95 <= 0.95
- mechanism falsified: LCI95 >= 1.00
- otherwise inconclusive

Static-ID equivalence is mandatory: the entire 95% CI of
`loss(static_id_permute) / loss(coreno_full)` must lie in `[0.98, 1.02]`.
If simple relabeling changes the result, the identity interpretation is considered invalid or implementation-sensitive.

### Run locally

```bash
pip install -r requirements.txt
python phase1b/run_phase1b.py --mode smoke --outdir results/phase1b
python phase1b/validate_phase1b.py --mode smoke \
  --raw results/phase1b/phase1b_smoke_raw.csv \
  --decision results/phase1b/phase1b_smoke_decision.json
```

The GitHub Actions workflow runs the smoke test first and, only if it passes, executes the frozen 200-seed confirmatory experiment and independent validator.

This is an engineering falsification study only. It makes no claim about CORENO ontology, consciousness, or liberation.
