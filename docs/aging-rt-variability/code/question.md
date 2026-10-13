# Question

## Original (user, 2026-10-08)
Does reaction time slow, or become more variable (coefficient of variation), with age?

## Current explication (draft, pending exploration)
In mice performing the IBL biased-block visual decision task, is age at session (days, continuous) associated with
(H1) a longer typical RT and/or (H2) higher RT variability, at the level of the individual mouse?

## Terms (draft; to be fixed during exploration)
- **Age**: `sessions.age_at_session_days`; per-mouse summary = mean over that mouse's included sessions.
- **RT**: response time = `response_times − stimOn_times` (the paper's definition; user choice 2026-10-08).
  Movement-onset RT (`firstMovement_times − stimOn_times`) is kept only as a secondary check.
- **Typical RT**: median RT per session.
- **RT variability** (candidates): CV = SD/mean; robust CV = IQR/median or MAD/median; computed per contrast
  then averaged, so differences in contrast mix do not leak in.
- **Trial inclusion** (primary, paper's rules; user 2026-10-08): first 400 trials; drop no-go trials and trials
  missing choice, probabilityLeft, feedbackType, feedback_times, stimOn_times or firstMovement_times; RT window
  0.08–2 s; trials with wheel movement before stimOn kept. RT window and early-movement handling are tested as
  grid parameters for robustness.
- **Tail rates** (added 2026-10-08): fraction of complete first-400 trials with RT < 0.08 s (anticipations) and
  with RT > 2 s (slow trials), as separate measures alongside median and spread.
- **Replicate / independent unit**: mouse (158). Sessions are nested within mice.

## Locked hypotheses (user approved 2026-10-08)
- H-var: RT variability, measured as SD of log RT per session, increases with age across mice (primary).
- H-slow (narrowed): median RT increases with age within the BWM cohort; no slowing across both cohorts.
- See `exploratory-analyses/004_locked_test_proposal.md`.

## Known design limits
- Age is confounded with cohort (bwm 92–456 d, lifespan 318–597 d), and with lab, recording year and rig.
- Age ranges overlap only at 317–461 d (7 bwm mice, 5 lifespan mice).
- Within-mouse ageing is not testable: median within-mouse age span is 3 days (bwm); 3 mice span ≥ 30 days.
- The paper (Zang et al. 2026) analysed all of these sessions and reports more variable RTs in older mice, so
  this is a re-analysis whose hypothesis is not naive with respect to the data.

## Data
`ibl_aging_zang2025` 1.0.0 (resolved via `data_locations.local.yaml`), tables
`trials`, `sessions`, `subjects`.

## Exploration / confirmation split
By mouse, 30% exploration / 70% confirmation (user choice 2026-10-08). BWM mice stratified by lab, lifespan mice
stratified as one group; fixed seed. Assignment in `artifacts/split.csv`.
