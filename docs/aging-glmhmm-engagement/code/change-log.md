# Change log

- 2026-10-12: created `question.md` (H1–H4, draft terms, proposed reuse of the `aging_rt_variability` split) and
  `TODO.md` (steps 0–13).
- 2026-10-12: TODO steps 0–3 marked done with outputs (no plan changes).
- 2026-10-12: question.md H2 redefined as engaged vs disengaged bouts (biased states not identifiable in the smoke
  test). TODO step 4 done; step 5 rewritten: CV by held-out trial chunks instead of held-out sessions (lifespan mice
  have 1–3 sessions), scores the full global → per-mouse procedure, two-stage reduced grid (user approved).
- 2026-10-12: step 5 plan: transition prior dropped (NeMoS 0.2.9 accepts dirichlet_transition_proba but never passes
  it to the M-step, so all three prior settings gave identical fits); Ridge grid extended to 0.01 and 0.03 because
  0.1 was best at the grid edge. Per-mouse transitions are now unregularised MLE. User approved.
- 2026-10-12: TODO step 5 done (K = 3, Ridge 0.1, raw contrast); 'engaged' = union of the two high-stimulus states.
- 2026-10-12: question.md: engaged-state and K definitions updated to match step 5.
- 2026-10-12: TODO step 6 done; occupancy metric found unreliable (see 004_metric_risk_note.md).
- 2026-10-12: K changed from 3 to 2 with per-mouse state labels (user choice after step 6 reliability results);
  added TODO step 6b (`005_k2_individual_fits.py`). glmhmm_metrics.py default K = 2; 004 pinned to K = 3.
- 2026-10-12: question.md: added Related work (Mohammadi, Ashwood & Pillow 2025) and reward-history caveat for H1.
- 2026-10-12: TODO step 6b done (see 005_metric_risk_note.md).
- 2026-10-12: TODO step 7 done (user approved step-8 setup); step 8 script renamed 006_. question.md: H1 measured by
  soft occupancy; engaged stimulus weight added as co-primary; H2 descriptive only.
- 2026-10-12: TODO step 8 done (exploration age effects; see 006_age_effects_exploration.png).
- 2026-10-12: added TODO step 8b (model-free early-trial check, user suggestion).
- 2026-10-12: locked-test proposal: added add-on model-free test C3 (easy-trial accuracy, first 90 trials) outside the primary sequence (user request).
- 2026-10-12: TODO step 9 done; tests locked (006_locked_test_proposal.md, glmhmm_metrics.py sha256 a11a42e7f6664355).
- 2026-10-12: TODO step 10 done (confirmatory tests run).
- 2026-10-12: added TODO step 10b (post hoc checks, user request; kept separate from C1–C3).
- 2026-10-12: question.md: corrected unverified claim that Zang et al. report more lapses.
- 2026-10-12: TODO steps 11 and 13 done; step 12 (neural extension) skipped for now.
- 2026-10-12: report original-question section reworded at user request (motivation from Ryan et al. kept as a note).
- 2026-10-12: report original question: removed 'simple analysis' note; added that the user directed the GLM-HMM (Ashwood et al. 2022) with NeMoS.
- 2026-10-12: report corrected: Zang et al. described as 'more variable' only; age-matched occupancy comparison stated as inconclusive (5 vs 6 mice, overlapping ranges), stimulus following as more consistent.
- 2026-10-12: TODO step 10c (timing diagnostics + trial-filter sensitivity, user request); report post hoc, plan (trial inclusion statement) and discussion updated.
- 2026-10-12: report edits (user): lab wording softened; filter disagreement explained; post hoc tail of inaccurate lifespan mice added (4 of 5 overlap weak engaged states); model-free lesson qualified; Zang/RT-variability contradiction fixed.
- 2026-10-12: report post hoc section trimmed and restructured (user request); Fig 10 cropped to panels 1-3 as confirmatory-analyses/010_posthoc_figure_panels1-3.png (image crop, no code change); timing figure moved to supplementary; H3 lapse decomposition removed; sessions moved to Introduction; Discussion aligned and tail sentence added.
- 2026-10-13: added TODO step 10d (weak-engaged-state exclusion check for C1a, user request after report review).
- 2026-10-13: report review fixes (user request; title unchanged): age-matched comparison worded symmetrically
  (both measures inconclusive); stimulus-following p-values in the sensitivity table marked descriptive (C1b not
  evaluated); weak-state exclusion table added; tail text moved out of the sensitivity caption, with a note that the tail
  includes the non-churchlandlab lifespan mouse; locked plan notes the test-order error and C3 multiplicity; ICC values
  made consistent (0.39 sessions / 0.50 halves); Ryan et al. intro limited to what the abstract states (Zang et al.
  'slightly worse' verified). `010_posthoc_figure.py` now writes the 3-panel report figure directly (replacing the
  image crop) with descriptive panel titles and a legend for the reference lines; no change to plotted data.
- 2026-10-13: report correctness/readability review (user request; title and original question unchanged): Zang-filter
  time-engaged p corrected (1e-4 → 5e-5, text and table); satiety, RT-variability and tail claims hedged (tail
  "clearest cohort difference" removed); C3 conclusion states the filter dependence; stimulus-following difference
  in the Discussion marked descriptive; Discussion split into bullets; definitions added (block prior, posterior,
  stimulus weight, lapse proxy, ICC, quiescence timer, precondition threshold, K = 3 seed dependence); "churchlandlab"
  → Churchland lab; state-comparability sample (all 158 mice, two global fits) stated; trials-kept column marked
  per-mouse median; df removed from permutation test row. Note: TODO 10b's "age-matched lifespan still below BWM"
  was superseded by the inconclusive wording in the report (2026-10-12).
- 2026-10-13: Discussion opening reworded (user request): dropped 'simplest analysis suggested by Ryan et al.'.
