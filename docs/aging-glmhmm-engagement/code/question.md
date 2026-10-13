# Question

## Original (user, 2026-10-12)
Which simple analysis suggested by Ryan, Ye & Churchland (2026, arXiv:2603.02461, "Understanding Decision-Making
Across the Lifespan Needs Theoretical Neuroscience") can be done on the aging dataset? Chosen: latent-state
(GLM-HMM) modelling of choices, following Ashwood et al. (2022, Nat Neurosci 25:201–212).

## Current explication (draft, pending exploration)
In mice performing the IBL biased-block visual decision task, does age change how mice move between latent
engagement states, and does that explain their higher lapse rates?

- **H1 (primary):** engaged-state occupancy decreases with age. Occupancy measured as soft occupancy (mean
  P(engaged) over trials; user choice 2026-10-12, slightly more reliable than the 0.5 threshold).
- **H-engaged (co-primary, added 2026-10-12):** the per-mouse engaged-state stimulus weight decreases with age, i.e.
  performance while engaged declines (the alternative to H3). Most reliable metric (ICC 0.76–0.86).
- **H2 (revised 2026-10-12; descriptive only since step 6b, unreliable):** older mice have longer disengaged bouts and fewer switches between engaged and
  disengaged trials (switches per 100 trials). Defined on engaged vs not-engaged only, because the smoke test showed
  the two biased states are not separately identifiable (mouse analogue of human MEG/fMRI findings cited by Ryan et al.).
- **H3 (mechanism of lapses):** the age-related rise in lapse rate is explained by time spent in disengaged
  states, not by worse engaged-state performance. Once occupancy is accounted for, age no longer predicts lapse
  rate, and the engaged-state stimulus weight doesn't change with age.
- **H4 (candidate, links to `aging_rt_variability`):** does disengaged-state occupancy account for the confirmed
  age-related increase in RT variability (SD of log RT)?

Alternatives we can tell apart: if occupancy doesn't change but the engaged-state weights fall with age, the
decline is in sensory or decision quality; if both change, both are involved; if neither changes, the higher
lapse rate comes from lapsing evenly across trials.

## Terms (draft; to be fixed during exploration)
- **Age**: `sessions.age_at_session_days`; per-mouse summary = mean over that mouse's included sessions.
- **GLM-HMM**: `nemos.glm_hmm.GLMHMM` (NeMoS 0.2.9), Bernoulli choices (right vs left), one logistic GLM per state,
  fit by EM; HMM reset at each session start (`session_starts`).
- **Inputs (per trial)**: signed contrast (contrastRight − contrastLeft, transform to be chosen), previous choice,
  previous reward × previous choice (win-stay/lose-switch), block prior (`probabilityLeft`; added because our
  sessions use biased blocks, unlike Ashwood's 50:50 fits), intercept (bias).
- **Number of states K**: K = 3 (2026-10-12), plateau of cross-validated per-mouse log-likelihood on held-out
  trial chunks; Ridge 0.1; raw signed contrast. Transition prior not applied (NeMoS 0.2.9 ignores it).
- **Engaged state** (revised 2026-10-12): the union of states whose global-fit stimulus weight is > 5 (with K = 3,
  two high-stimulus states differing in bias); disengaged = the low-stimulus state. P(engaged) = summed posterior.
- **Occupancy**: soft occupancy = mean smoothed P(engaged) over a mouse's trials (primary); fraction with
  P(engaged) > 0.5 kept as a check.
- **Dwell time**: from the fitted transition matrix, checked against empirical run lengths of decoded states.
- **Lapse rate**: as in `aging_rt_variability` / Zang et al. (psychometric lapse); definition to be fixed.
- **Fitting**: global fit on all exploration mice, then per-mouse fits started from it (`init_params`), with a
  Dirichlet prior on transitions favouring persistence and a Ridge penalty on weights. NeMoS has no hierarchical
  weight prior and no covariate-driven transitions, so age is tested in a second stage on per-mouse metrics.
- **Replicate / independent unit**: mouse (158). Sessions are nested within mice.

## Known design limits
- Age is confounded with cohort (bwm 92–461 d, lifespan 317–597 d), lab, recording year and rig.
- Within-lab contrast: churchlandlab bwm (11 mice, 140–270 d) vs lifespan (18 mice, 317–597 d).
- Cross-sectional only: lifespan mice have 1–3 sessions; within-mouse ageing isn't testable.
- Lifespan sessions have ~400–1500 trials per mouse; per-mouse state estimates may be noisy.
- Zang et al. (2026) analysed all of these sessions and report slightly worse performance and more variable RTs in older mice (a lapse-specific claim was not verified); the confirmation mice
  were already used for the RT tests in `aging_rt_variability`, so H4 is not naive with respect to those data.

## Related work (checked 2026-10-12)
- Mohammadi, Ashwood & Pillow (2025, Nat Commun, doi:10.1038/s41467-025-66738-0; bioRxiv 2024.02.02.578482, with
  IBL): on IBL biased-block data, a 4-state GLM-HMM with input-driven transitions (two engaged states with weak
  left/right bias, two disengaged states with strong bias; bias follows the block). Past choices and stimuli predict
  left/right-bias switches; past rewards predict switches to disengagement (possibly satiety).
- Bearing on this project: explains why our K = 3 alternated between a biased engaged and a biased disengaged
  solution; K = 2 collapses their four states into engaged vs disengaged. Caveat for H1: an age difference in
  occupancy could reflect motivation/satiety (reward history) rather than ability; check in step 8.

## Data
`ibl_aging_zang2025` 1.0.0 (resolved via `data_locations.local.yaml`), tables
`trials`, `sessions`, `subjects`; spikes shards only for the optional neural extension.

## Exploration / confirmation split (proposed, pending user approval)
Reuse the split from `aging_rt_variability` (`projects/aging_rt_variability/artifacts/split.csv`): by mouse, 30%
exploration / 70% confirmation, BWM stratified by lab, lifespan as one group, fixed seed.
- exploration: 42 bwm + 6 lifespan mice (churchlandlab: 3 bwm + 6 lifespan)
- confirmation: 97 bwm + 13 lifespan mice (churchlandlab: 8 bwm + 12 lifespan)
