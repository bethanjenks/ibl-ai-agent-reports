# Metric risk note: per-mouse GLM-HMM engagement metrics (step 6)

Settings: K = 3, Ridge 0.1, raw contrast, global fit (best of 3 restarts) on 48 exploration mice, then per-mouse fits
started from it. Figure: `004_individual_fits.png`. Reliability = Spearman ρ across mice.

| metric | other global seed | even vs odd sessions (n = 36) | first vs second half of sessions | K = 2 vs K = 3 |
| --- | --- | --- | --- | --- |
| occupancy (P > 0.5) | 0.42 | 0.35 | 0.60 | 0.72 |
| soft occupancy | 0.44 | 0.43 | 0.60 | 0.79 |
| disengaged bout length | 0.42 | 0.13 | 0.47 | 0.63 |
| switches per 100 trials | 0.44 | 0.39 | 0.12 | 0.59 |
| engaged stimulus weight | 0.95 | 0.81 | 0.88 | 0.91 |

## What the diagnostics support
- The full-data global fit has Ashwood's structure: one engaged state (stimulus weight 16.5) and two low-stimulus
  states biased right and left. Threshold and soft occupancy agree closely (panel e).
- The per-mouse engaged-state stimulus weight is highly reliable across seeds, sessions and K.

## What they do not support
- **Occupancy depends on which global optimum is found (ρ = 0.42 across seeds).** In the CV folds and the other seed,
  the global fit often finds two stimulus-driven states (one biased) instead of two biased low-stimulus states. Whether
  the biased-but-stimulus-driven mode counts as "engaged" then changes with the optimum, moving most mice's occupancy
  up to 0.8–1.
- **State identity is not preserved in some per-mouse fits** (panel c): several mice's "biased" states end up with
  stimulus weights of 9–13, and some lifespan mice's "engaged" state has a weight of 2–7. The label assigned from the
  global fit then no longer means the same thing in every mouse.
- **Bout length and switch rate are unreliable** (ρ 0.12–0.47) and should not be primary metrics.
- Example CSHL067: the first ~520 trials of a session are called disengaged although errors are spread throughout,
  so a session-long bias can be absorbed as a "disengaged" state. The metric may partly track session-level bias
  rather than trial-by-trial engagement.

## Interpretations
- Technical: multiple near-equivalent optima and label drift (most likely, given the seed result).
- Biological: engagement may genuinely vary from day to day, which would also lower the session split-half ρ (0.35);
  the within-session split (0.60) is higher.

## Recommendation
Pause before the age analysis. Make occupancy robust to the identity problem, then re-check reliability:
1. label engaged states per mouse (per-mouse stimulus weight > threshold) rather than from the global fit; and/or
2. use K = 2 (engaged vs disengaged; global stimulus weights 14.5 vs 2.2), which has fewer competing optima.
Engaged-state stimulus weight is reliable already and can carry H3.
