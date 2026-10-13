# Question

## Original (user, 2026-10-13)
Do older mice show weaker quenching only because they are disengaged more often, or also when they are engaged?

## Background
- Zang et al. (2026, Nat Commun, doi:10.1038/s41467-026-74227-1) report that stimulus-induced Fano-factor (FF)
  quenching is attenuated with age (strongest in VISp+pm, MOs, LS/ACB, LP), using all of the first 400 trials.
- `aging_glmhmm_engagement` (this repo) found with a K = 2 GLM-HMM that lifespan (older) mice spend less time in
  the engaged state (confirmed, C1a) and that their engaged state is weaker (lower stimulus weight). Per-trial
  P(engaged) exists for all 158 mice (`artifacts/005_posteriors.parquet`, `008_posteriors_confirmation.parquet`).
- The age contrast is effectively cohort (lifespan 317–597 d vs bwm 92–461 d); no graded age trend in engagement
  was seen within bwm.

## Current explication (draft, pending exploration)
Metrics (user, 2026-10-13): pre- and post-stimulus firing rate (FR), pre- and post-stimulus Fano factor (FF), and
FF quenching (post − pre). Two parts:
- **Part A, state:** within a neuron, compare each metric on engaged vs disengaged trials (trial-count matched).
  Expected from the original question: weaker quenching on disengaged trials.
- **Part B, age on engaged trials:** do the age effects reported by Zang et al. (higher FR overall, higher post-FF,
  weaker quenching; pre-FF unaffected) still hold using engaged trials only?
- Part A is descriptive only (user, 2026-10-13: underpowered, effects within drift placebo); confirmatory tests
  are for Part B. Part B metrics: paper windows (primary, comparable to Zang et al.) plus a movement-safe variant.
- **Mediation (descriptive):** Δ = β_all − β_eng per metric (equal trial count, same neurons), mouse-bootstrap CI.
- Tested (proposal 004 + amendment): quench C1–C3 (primary); fr_pre, fr_post, ff_post, ff_pre F1–F4 (secondary).

Possible outcomes:
- age effect on engaged-only quench ≈ all-trial effect → not explained by disengagement;
- engaged-only effect vanishes and disengaged trials quench less → explained by disengagement;
- in between → both.

## Terms (draft; to be fixed during exploration)
- **FR**: mean count / 0.1 s in the pre and post windows (same windows as FF).
- **Quench index** (paper definition, starting point): mean-subtracted FF (residuals after subtracting the mean
  count per signed-contrast condition, 9 conditions) in 100 ms windows aligned to `stimOn_times`;
  post [0.16, 0.26] s minus pre [−0.10, 0] s. Negative = quenching. Windows to be revisited in exploration.
- **Engaged / disengaged trial**: P(engaged) > θ / < 1 − θ from the existing K = 2 GLM-HMM posteriors; θ = 0.8
  starting value. Note that the engaged state in older mice is itself weaker (lower stimulus weight).
- **Trial-count matching**: FF estimates depend on trial count, so engaged-only and all-trial quench are computed
  on the same number of subsampled trials per neuron (repeated random subsamples).
- **Trials**: paper filters (choice, probabilityLeft, feedback, stimOn, firstMovement present; RT 0.08–2 s).
  All trials (user 2026-10-13); the paper's first-400 cut is a sensitivity check.
- **Neurons**: label == 1 (package), firing rate > 1 Hz and presence ratio > 0.95 in the task window
  (`task_firing_rate`, `task_presence_ratio`), Beryl ROIs from the paper.
- **Age**: `sessions.age_at_session_days`. **Independent unit**: mouse; neurons nested in insertions, sessions, mice.

## Known design limits
- Age = cohort, lab, year, probe model, regions sampled (see dataset `scientific-context.md`).
- Within-lab contrast: churchlandlab bwm (11 mice, 140–270 d) vs lifespan (18 mice).
- bwm sessions have few disengaged trials (exploration median 16 with P < 0.2), so disengaged-trial FF is only
  estimable in a subset of sessions, mostly lifespan.
- Engagement is concentrated late in sessions: time-in-session drift (firing-rate nonstationarity) can mimic a
  state effect.
- churchlandlab mice move early more often; movement inside the post window can change FF.
- Paper analysed all these sessions, so the age effect on all-trial quench is not naive; the engaged-only
  question is.

## Data
`ibl_aging_zang2025` 1.0.0 (resolved via `data_locations.local.yaml`): `trials`,
`sessions`, `units`, spike shards. GLM-HMM posteriors from `projects/aging_glmhmm_engagement/artifacts/`.

## Exploration / confirmation split (approved 2026-10-13)
Reuse `aging_glmhmm_engagement` split (by mouse, 30/70, bwm stratified by lab): 48 exploration mice (42 bwm + 6
lifespan), 110 confirmation (97 bwm + 13 lifespan). Posteriors were fitted separately on each split, so the
confirmation engagement labels did not see exploration data.
