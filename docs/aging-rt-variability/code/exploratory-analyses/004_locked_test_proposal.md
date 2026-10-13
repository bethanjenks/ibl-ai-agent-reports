# LOCKED confirmatory tests (approved by user 2026-10-08)

Data: confirmation mice only (110: 97 BWM, 13 lifespan), `artifacts/split.csv`.
Trial rules (primary): first 400 trials; complete (no no-go, required events present); RT = response − stimOn in
0.08–2 s; early-movement trials kept. Metrics from `exploratory-analyses/rt_metrics.py` (frozen; sha256 prefix a4e5c943888c05f2).
Mouse-level value = mean over the mouse's sessions; mouse age = mean age at session.

## Primary
**P1. RT variability increases with age.** Spearman ρ(mouse age, mouse sd_log_rt) > 0, all 110 mice.
One-sided permutation test (10,000 shuffles of age across mice, seed 0), α = 0.05.
Exploration: ρ = +0.45 (95% bootstrap CI +0.14 to +0.70), n = 48.
Power (Fisher z, one-sided α = 0.05, n = 110): 0.94 if true ρ = 0.30; 0.84 at 0.25; 0.67 at 0.20 (BWM only, n = 97: 0.91, 0.80, 0.63).

## Secondary (one-sided, Holm-corrected across S1–S4)
- **S1. Not only a cohort effect:** P1 restricted to the 97 BWM mice. Exploration ρ = +0.38.
- **S2. Not a lab effect; sessions nested in mice:** session-level mixed model
  `sd_log_rt_z ~ age_z + C(lab) + (1 | mouse)`, β_age > 0. Exploration β = 0.45 (CI 0.16–0.74).
- **S3. Paper's metric:** P1 with CV (SD/mean) in place of sd_log_rt. Exploration ρ = +0.35.
  Caveat: CV changes sign when the upper RT cut is raised to 5 s or removed (`003_parameter_grid.png`), so a
  positive result holds only for the 0.08–2 s window.
- **S4. Slowing within BWM:** Spearman ρ(age, median RT) > 0 in the 97 BWM mice. Exploration ρ = +0.31,
  lab-centred +0.54; across both cohorts ρ = +0.12 because lifespan mice are fast.

## Descriptive only (no test)
Median RT across both cohorts; tail rates (RT < 0.15 s, RT > 1 or 2 s); grid sensitivity of P1 repeated on the
confirmation set; overlap-window (300–460 d) cohort comparison.

## Adversarial review: ways the hypothesis is true but the test fails
1. **Exploration ρ inflated by choosing the best of five spread metrics.** Expect shrinkage; power stays
   ≥ 0.84 down to ρ = 0.25. sd_log_rt was also the most stable metric across the grid (positive in 48/48
   settings), not just the peak.
2. **Window choice.** sd_log_rt is positive in every setting with an upper cut ≥ 2 s and a lower cut 0–0.15 s, so
   the locked window is not a knife-edge. CV is a knife-edge, so it is secondary only.
3. **Lifespan heterogeneity.** One exploration lifespan mouse (≈ 545 d) has the lowest variability of all
   mice. If several confirmation lifespan mice are like it, P1 weakens. S1 (BWM only) does not depend on them.
4. **Lab mix.** Lab proportions match between sets by construction; S2 adjusts for lab.
5. **Age co-varies with training duration and experience** (paper Suppl. Fig. 2). This affects interpretation,
   not test validity; training history is not in the package.
6. **Cohort/lab/rig/year all co-vary with age in the lifespan cohort.** P1 cannot separate them; S1 and the
   lab-centred exploration result (ρ = +0.42) are the evidence that the effect exists within BWM too.
