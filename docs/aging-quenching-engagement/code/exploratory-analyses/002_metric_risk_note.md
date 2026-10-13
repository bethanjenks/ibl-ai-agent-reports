# Metric risk note (step 3 pilot, exploration mice, 2026-10-13)

Source: `002_ff_pilot.{py,png}`, `001_ff_single_session.png`, `artifacts/002_*`.

## What the diagnostics support
- FR, mean-subtracted FF and quench compute sensibly; population FF falls after stimOn in the lifespan example session.
- All trials vs first 400 trials give nearly identical session-median quench (keeping all trials costs nothing).
- Part A (engaged vs disengaged) can be estimated in only 20 bwm + 7 lifespan exploration sessions at θ = 0.8
  (≥ 20 disengaged trials); 45 + 6 at θ = 0.6; 14 + 4 at θ = 0.9.

## Problems found
1. **FF depends on trial count**: median engaged FF rises from 1.22 to 1.31 (bwm, pre) between 20 and 250 trials,
   even after the n − k correction. More trials span more session time, so slow drift adds variance. Every
   comparison must use equal trial counts; for Part A also comparable time spans.
2. **Time in session**: bwm disengagement is mostly a block at session end; lifespan disengagement is interleaved
   mid-session. The early-vs-late engaged placebo (no state change) moves FR by ~1 Hz and FF/quench by as much as
   the state contrast does. Part A differences are within the drift noise.
3. **Movement inside the post window**: median first movement is 0.14 s (bwm) and 0.12 s (lifespan) after
   stimOn; 59% / 66% of trials move before the post window [0.16, 0.26] s starts and 84% / 85% before it ends.
   The paper's post-FF and quench therefore largely measure the movement-onset period. Equal across states
   (so Part A is not affected), but lifespan mice move earlier (28% vs 15% before 40 ms), so it does matter for
   Part B (age). An earlier post window (e.g. [0.04, 0.14] s) or excluding trials that moved before the post
   window ends is a candidate alternative.

## Preliminary Part A (θ = 0.8, session medians of within-neuron disengaged − engaged)
- quench: bwm +0.035 (less quenching on disengaged), lifespan −0.045 (MORE quenching on disengaged);
  time-matched control: bwm +0.10, lifespan −0.07. Opposite signs in the two cohorts; no consistent state effect.
- ff_pre, ff_post, fr: small differences in both directions, comparable to the placebo.
- If disengaged trials do not quench less in older mice, more disengagement cannot explain their weaker
  quenching, so Part B (engaged-only) is the decisive test.

## Interpretations not ruled out
- Part A is underpowered (7 lifespan sessions); a real but small state effect would be missed.
- The engaged state in lifespan mice is itself weaker (aging_glmhmm_engagement), so "engaged" differs by cohort.
- Region mix differs by cohort (lifespan: MOs, VISp+pm, MBm dominate); Part B must compare within ROI.

## Recommendation
Proceed to step 5 with: θ = 0.8 (θ = 0.6 as a check); Part B at N_B = 150 matched trials (engaged vs all);
quench windows paper default plus one movement-safe alternative, chosen before looking at age effects;
mixed model with ROI and mouse.
