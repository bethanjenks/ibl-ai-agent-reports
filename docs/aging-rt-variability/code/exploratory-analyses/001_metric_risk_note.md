# 001 RT metric risk note (exploration set: 48 mice, 148 sessions)

**Supported**
- `response_times` is a wheel threshold crossing: median |wheel displacement| from stimOn to response = 0.281 rad
  in every exploration session, both cohorts, all labs. The same threshold applies everywhere, so the RT end
  point is comparable across cohorts.
- Response RT ≥ movement-onset RT on almost all trials; the gap (movement-to-threshold time) is variable.
- RT is roughly stable across trial number; the first-400 cut does not select an unusual part of the session.

**Not ruled out / risks**
- The RT distribution is bimodal: a fast mode near 0.1 s, present at contrast 0 too, so these are not
  stimulus-driven responses (anticipations). The mode is bigger in lifespan mice, and the 0.08 s lower bound cuts
  through it (0.3% bwm, 0.9% lifespan removed).
- The slow tail is heavy and session-specific: RT > 2 s removes 11.2% of bwm and 7.1% of lifespan complete
  trials, up to 57% in one session.
- Wheel movement before stimOn: 12.8% of kept bwm trials and 19.4% of kept lifespan trials. For these, RT
  partly measures movement that started before the stimulus.
- So "variability" can rise with age through more fast anticipations, or fall through fewer slow trials,
  depending on how the tails are handled. A CV computed inside 0.08–2 s mostly sees the body of the
  distribution. The window choice is a parameter that can flip the sign of an effect.
- The age axis is cohort plus a sparse bwm tail (> 300 d). No age-matched comparison is possible at exploration
  scale (lifespan: 6 mice).
- Preview only: session median RT shows no obvious slowing with age; the oldest lifespan sessions are among the
  fastest.

**Decision**: proceed to variability-metric definition (step 4), treating the RT window and early-movement
handling as grid parameters (step 5).
