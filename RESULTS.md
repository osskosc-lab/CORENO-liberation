# Frozen Phase 1 result

The registered 100-seed run is **INCONCLUSIVE**.

| Metric | Result |
|---|---:|
| `Q = L(CORENO full) / L(strong)` | 0.941 |
| Paired bootstrap 95% CI | [0.924, 0.957] |
| Preregistered support bound | upper CI <= 0.900 |
| High-confidence wrong-rate delta | -0.021 |
| CORENO STOP rate | 0.007 |

CORENO-full improved the primary loss by about 5.9% relative to the strong
baseline and passed both safety gates. The registered claim required at least a
10% improvement with the 95% upper CI at or below 0.90, so this does not count
as support. It is not classified as falsified because the lower CI is below
1.00.

The explanatory checks do not establish the proposed full mechanism:

- Removing the plural-weight floor worsened primary loss by 0.016.
- Removing dissent changed the displayed primary loss by 0.000.
- Removing reconstruction changed the displayed primary loss by approximately
  0.000 (slightly lower at the shown precision).
- Shuffling reconstruction history had no displayed effect.
- Breaking the mapping between historical performance and agent identity
  worsened primary loss by 0.043.

The raw seed results and condition summary are included in `results/`.
The reproducible report is generated locally by `generate_report.py`.
