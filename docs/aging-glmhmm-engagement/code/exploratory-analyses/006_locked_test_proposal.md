# LOCKED confirmatory tests (approved by user 2026-10-12)

Data: confirmation mice only (110: 97 BWM aged 102–429 d, 13 lifespan aged 318–597 d), `artifacts/split.csv`.
All trials except no-go (`001_design_matrix.parquet`).
Model: K = 2 GLM-HMM, Ridge 0.1, raw signed contrast, inputs stim / previous choice / win-stay-lose-switch / block.
Global fit on the 110 confirmation mice (best of 3 restarts by training log-likelihood), then per-mouse fits started
from it; each mouse's engaged state = its higher-stimulus-weight state. Code: `exploratory-analyses/glmhmm_metrics.py`
(frozen; sha256 prefix a11a42e7f6664355), NeMoS 0.2.9.
Mouse-level metrics: soft occupancy = mean P(engaged) over the mouse's trials; engaged stimulus weight = the
per-mouse engaged-state stimulus weight. Mouse age = mean age at session.

Precondition (checked before any test, reported either way): the confirmation global fit has one state with
stimulus weight > 5 and one < 5. If not, stop and consult the user.

## Primary tests: fixed-sequence (gatekeeping), each at one-sided α = 0.05
Tests run in this order; a test is evaluated only if every earlier test was significant, which controls the
family-wise error rate at 0.05 without splitting α. Order follows expected power.

1. **C1a. Lifespan mice are engaged less than BWM mice.** Soft occupancy, lifespan (13) < BWM (97), one-sided
   Mann-Whitney U. Exploration: medians 0.60 vs 0.83, AUC 0.12 (n = 6 vs 42).
   Power (resampling exploration values, α = 0.05): ≈ 1.0; ≥ 0.99 if the effect is half as large.
2. **C2a. Within the lifespan cohort, older mice are engaged less.** Spearman ρ(age, soft occupancy) < 0 in the 13
   lifespan mice, one-sided permutation test (10,000 shuffles, seed 0). Exploration ρ = −0.60 (n = 6).
   Power (Fisher z, n = 13): 0.86 if true ρ = −0.7, 0.71 at −0.6, 0.54 at −0.5.
3. **C1b. Lifespan mice follow the stimulus less when engaged.** Engaged stimulus weight, lifespan < BWM, one-sided
   Mann-Whitney U. Exploration: medians 7.4 vs 12.1, AUC 0.20. Power ≈ 0.95; ≈ 0.53 if the effect is half as large.
4. **C2b. Within the lifespan cohort, older mice follow the stimulus less when engaged.** Spearman ρ(age, engaged
   stimulus weight) < 0, 13 lifespan mice, permutation test. Exploration ρ = −0.60.

## Add-on model-free test (user request 2026-10-12; separate from the primary sequence, one-sided α = 0.05)
Evaluated regardless of the primary results, so a failure in the model-based sequence does not block it.

5. **C3. Lifespan mice are less accurate on easy trials at the start of sessions.** Per-mouse accuracy on trials with
   |contrast| ≥ 0.25 among the first 90 trials of each session (an unbiased 50:50 block, so no block prior; no-go
   trials excluded), pooled over the mouse's sessions; lifespan (13) < BWM (97), one-sided Mann-Whitney U.
   Code: `exploratory-analyses/007_model_free_early_trials.py` (window 90). Exploration: medians 0.89 vs 0.95,
   AUC 0.13 (n = 6 vs 42). Power (resampling, α = 0.05): ≈ 1.0; 0.99 if the effect is half as large.
   Same lab/cohort confound as C1 (see review item 3). Reaction times in this window are descriptive only.

## Descriptive only (no tests)
- Linear age correlation across all 110 mice and within BWM only (exploration: ρ ≈ −0.2 and ≈ 0).
- churchlandlab-only lifespan (12) vs BWM (8) comparison: the only within-lab contrast.
- H3: lapse proxy ~ age, with and without occupancy and engaged stimulus weight (z-scored OLS).
- Reward-history (satiety) check, first-session-only and before-100th-reward occupancy.
- H2 bout length and switch rate vs age.

## Adversarial review: ways the hypothesis is true but a test fails
1. **Lifespan ages are clustered.** 10 of 13 confirmation lifespan mice are 519–597 d and 3 are 318–417 d, so C2
   rests largely on whether those 3 differ from the rest. If ageing effects saturate after ~500 d, C2 can fail even
   though ageing matters. C1 does not depend on this.
2. **Exploration effects come from 6 lifespan mice and are likely inflated.** C1a keeps power ≥ 0.99 at half the
   effect; C1b and C2 do not, which is why C1a is first.
3. **Lab and cohort confound C1.** 12 of 13 lifespan mice are churchlandlab (1 mrsicflogellab); BWM spans 12 labs,
   and the two cohorts differ in recording year and rig. A C1 result shows that the lifespan cohort differs, not that
   age is the cause. C2 is within one cohort (mostly one lab) and is the age test; the churchlandlab-only contrast is
   reported descriptively.
4. **Occupancy is only moderately reliable** (ICC 0.36–0.50 between sessions). Lifespan mice have 1–3 sessions, so
   per-mouse occupancy is noisy, which mostly reduces C2 power.
5. **The confirmation global fit could differ** (different mice). K = 2 was identical across seeds in exploration;
   the precondition guards against a degenerate fit.
6. **Ceiling.** Many BWM mice are near 0.9–1.0 occupancy; this helps C1 (lifespan below) but means no graded age
   trend can be seen within BWM, consistent with exploration.
7. **Motivation vs ability.** Lower occupancy could reflect motivation rather than cognitive ageing; exploration
   showed lifespan mice were less engaged from the start of sessions (not satiety), but water restriction or weight
   are not in the package.
