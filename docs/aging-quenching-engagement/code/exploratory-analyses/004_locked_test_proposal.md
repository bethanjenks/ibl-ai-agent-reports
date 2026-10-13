# Proposed locked tests (draft for user review, 2026-10-13)

## Exploration result (44 mice: 38 bwm, 6 lifespan)
- **Engaged vs all trials (equal N):** every age slope is about the same on engaged trials only. For quench,
  β_eng / β_all = 0.93 (paper window, all labs), 1.13 (churchlandlab) and 0.71 (movement-safe).
- **Part A:** disengaged trials do not quench less in lifespan mice (−0.045, opposite sign to bwm).
- Disengagement therefore does not account for the age effect.
- **Age effects themselves are weak at the mouse level.** Paper-direction signs for fr_pre, fr_post and quench;
  none for ff_pre or ff_post. Engaged-trial z (β / permutation SD):

  | metric | all labs, paper | all labs, safe | churchlandlab, paper |
  | --- | --- | --- | --- |
  | quench | 0.98 | 1.33 | 1.71 |
  | fr_pre | 1.44 | 1.38 | 0.63 |
  | ff_post | 0.37 | 0.36 | −0.26 |

  Zang et al. used neurons as the unit; we permute across mice, which is wider.

## Proposed tests (confirmation mice, engaged trials only, P(engaged) > 0.8, equal N)
Statistic: β_age, the OLS slope on mouse mean age with ROI fixed effects. One-sided (paper direction) p from 10,000
permutations of age across mice. Holm correction across C1–C3 at α = 0.05.

- **C1:** quench, paper windows, N = 150, churchlandlab only (within-lab bwm vs lifespan). β > 0.
- **C2:** quench, paper windows, N = 150, all labs. β > 0.
- **C3:** quench, movement-safe post window [0.02, 0.12] s, first movement > 0.12 s, N = 100, all labs. β > 0.

Descriptive (no p-values), with 95% mouse-bootstrap CIs:
- β_eng / β_all for all 5 metrics × 2 windows;
- β_age for fr_pre, fr_post, ff_pre and ff_post on engaged trials;
- Part A (dis − eng, time-matched, placebo).

Interpretation rule (fixed in advance):
- "Also when engaged" is supported if any of C1–C3 is significant after Holm correction.
- "Not explained by disengagement" is supported if, in addition, the C2 ratio's CI excludes 0.5.

## Adversarial review: how the hypothesis could be true and the tests still fail
1. **Power.**
   - Confirmation has 110 mice (2.5× exploration), so SE shrinks by about 0.63×.
   - Projected z: C1 ≈ 2.5 (power ~80%), C2 ≈ 1.6 (~50%), C3 ≈ 2.1 (~65%).
   - C1 sees only 20 mice; one outlying mouse can dominate it.
2. **Age = cohort.**
   - Lifespan has only 13 confirmation mice.
   - The engaged state is weaker in lifespan mice (aging_glmhmm_engagement), so "engaged" is not equivalent
     across ages.
3. **Region mix.**
   - Lifespan has no PO or ORB, and few CP, LP or DG units.
   - ROI fixed effects absorb level differences, not age × ROI.
   - The paper's strongest regions (VISp+pm, MOs, LS, ACB, LP) are partly missing.
4. **Movement.** The paper-window quench is mostly movement-locked (003). C3 is the movement-safe check, but it
   keeps fewer trials.
5. **Exploration choices.**
   - θ = 0.8, N = 150/100 and E = 0.12 were chosen on exploration data. E was chosen on bwm only and is blind to
     age.
   - Confirmation posteriors come from the separate confirmation GLM-HMM fit.
6. **Non-independence.** Zang et al. analysed these same sessions, so all-trial age effects are not new evidence.
   Only the engaged-only comparison is new.

## Alternatives considered
- **Lifespan-vs-bwm contrast instead of the age slope:** about as powerful, but further from the paper.
- **Neuron-level mixed model (paper-like):** more power, but the unit of replication would be wrong.

## Amendment (2026-10-13, user: "add firing rate as a tested metric and fano factor")
Secondary family F (Holm across F1–F4, α = 0.05). Engaged trials, paper windows, N = 150, all labs, ROI fixed
effects, mouse permutation. Primary family Q (C1–C3) is unchanged and corrected on its own.

| test | metric | alternative | exploration z | projected power |
| --- | --- | --- | --- | --- |
| F1 | fr_pre | β > 0 (paper: higher FR with age) | 1.45 | ~73% |
| F2 | fr_post | β > 0 | 1.10 | ~55% |
| F3 | ff_post | β > 0 (paper: higher post FF with age) | 0.37 | ~15% |
| F4 | ff_pre | two-sided (paper: no effect) | −0.35 | n/a |

- F4 cannot confirm "no effect"; a non-significant result is only consistent with the paper.
- The paper reports lower FR in thalamus. ROI × age is not tested.

## Amendment: engaged-vs-all criterion (code check on exploration mice)
- The ratio β_eng / β_all is unstable when β_all's CI includes 0. On exploration mice, C2's ratio CI was −1.13 to
  2.48, so the "ratio CI excludes 0.5" rule could essentially never be met.
- Replaced by Δ = β_all − β_eng, the part of the age slope that disappears when disengaged trials are removed.
  Δ is estimated with a 95% mouse-bootstrap CI.
- Δ is precise because both trial sets come from the same neurons. Exploration C2: Δ = 0.001 [−0.005, 0.010],
  β_all = 0.016.
- New rule: "not explained by disengagement" if Δ's upper CI bound is < 0.5 × β_all (C2). Exploration: 0.010 vs
  0.008, which narrowly fails; the CI should narrow by ~0.63× with 110 mice.
- Ratios are still reported.

Code: `confirmatory-analyses/005_confirmatory_tests.py`.
- Run on exploration mice as a check: it reproduces the step-5 slopes exactly.
- Check outputs carry the `_exploration_check` suffix.
- Runtime on exploration: 18 s once metrics are cached, about 1.5 min of metric computation before that.
- Confirmation (about 2.5× the sessions): about 5 min.
