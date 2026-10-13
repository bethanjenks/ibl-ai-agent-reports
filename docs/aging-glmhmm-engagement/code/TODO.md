# TODO: age and latent engagement states (GLM-HMM)

## Planning
- [X] Read Ryan et al. 2026; choose GLM-HMM; install NeMoS 0.2.9 (+ pynapple, JAX) outside the lockfile;
      draft question.md
- [X] 0. **Consult user:** approve plan, split strategy and consultation frequency
      (user: reuse RT split; check in at marked steps)
- [X] 1. `000_data_split.py`: copy the `aging_rt_variability` split to `artifacts/split.csv`; plot age/lab/cohort
      balance and trials per mouse for each split
      → `exploratory-analyses/000_data_split.{py,png}`, `artifacts/split.csv`

## Exploratory analysis (exploration mice only)
- [X] 2. `001_design_matrix.py`: per-trial inputs (signed contrast, previous choice, previous reward × choice,
      block prior, intercept) and choices for exploration sessions; save to `artifacts/`. Diagnostics:
      psychometric curves by cohort and block; no-go trials; whether to cut late trials (session-end
      disengagement is a real state, so keep all trials by default and test the first 400 as an alternative)
      → `exploratory-analyses/001_design_matrix.{py,png}`, `artifacts/001_design_matrix.parquet` (all mice)
- [X] 3. `002_nemos_smoke_test.py`: (a) simulate from a known 3-state GLM-HMM with realistic trial counts and
      check that NeMoS recovers the weights, transitions and occupancy; (b) fit 2–3 real sessions; time one fit
      and estimate the full run time
      → `exploratory-analyses/002_nemos_smoke_test.{py,png}`, `artifacts/002_real_occupancy_pilot.csv`
        (engaged occupancy recovered well, biased states not separated; real pilot: lifespan occupancy lower;
        ~25-40 s per global + 12-mouse fit on 23k trials, 8 CPUs, no GPU)
- [X] 4. **Consult user:** inputs, trial inclusion, recovery results, run-time estimate
      (user: keep all trials; H2 redefined as engaged vs disengaged bouts; reduced step-5 plan on 4 CPUs)
- [X] 5. `003_model_selection.py`: cross-validation by held-out contiguous trial chunks (5 per session, so mice
      with one session can be scored), scoring the full procedure (global fit → per-mouse fits); stage 1: K = 3,
      Ridge {0.1, 0.5, 2} × transition prior {flat, alpha 2, sticky}; stage 2: K = 1–4 and raw vs tanh contrast at
      the best setting; held-out log-likelihood overall and per cohort; rule for labelling states
      (2026-10-12: transition prior dropped, NeMoS 0.2.9 ignores it; Ridge extended to 0.01, 0.03; user approved)
      → `exploratory-analyses/003_model_selection.{py,png}`, `artifacts/003_cv/*.json`, `artifacts/003_cv_per_mouse.csv`,
        `artifacts/003_stage1.log` (Ridge 0.1, raw contrast, K = 3: plateau of per-mouse held-out LL; two
        high-stimulus states + one low-stimulus state; engaged = states with global stim weight > 5)
- [X] 6. `004_individual_fits.py`: per-mouse fits started from the global fit; metric-definition figure
      (posterior traces, occupancy, dwell times, engaged weights); reliability checks: random restarts,
      split-half by session, dependence on trial count
      → `exploratory-analyses/004_individual_fits.{py,png}`, `glmhmm_metrics.py` (shared), `004_metric_risk_note.md`,
        `artifacts/004_*` (occupancy unreliable: ρ 0.42 across global seeds, 0.35 across sessions, state-label drift;
        engaged stimulus weight reliable, ρ 0.81–0.95)
- [X] 6b. `005_k2_individual_fits.py`: K = 2 (engaged vs disengaged) with per-mouse state labels (engaged = each
      mouse's highest-stimulus-weight state); same reliability checks as step 6, plus K = 2 vs K = 3 occupancy
      (user choice after step 6: K = 3 alternates between solutions, so occupancy was unreliable)
      → `exploratory-analyses/005_k2_individual_fits.{py,png}`, `005_metric_risk_note.md`, `artifacts/005_*`
        (stable across seeds; occupancy ICC 0.36–0.50 between sessions/halves; engaged stim weight ICC 0.76–0.86)
- [X] 7. **Consult user:** K, metrics and their reliability
      (user: K = 2; H1 = soft occupancy; engaged stimulus weight co-primary; H2 descriptive; checks for
      churchlandlab-only, sessions per mouse and reward history)
- [X] 8. `006_age_effects_exploration.py`: occupancy, dwell time, transition rate, engaged weights and lapse vs
      age; churchlandlab within-lab contrast and all-lab mixed model; H3 lapse decomposition; H4 link to RT
      variability. Adversarial checks: block-prior misspecification, session length, trial position, cohort,
      sessions per mouse, end-of-session disengagement
      → `exploratory-analyses/006_age_effects_exploration.{py,png}`, `artifacts/006_age_stats_exploration.csv`
        (no graded age trend within BWM; lifespan mice lower occupancy and engaged weight than BWM; within the 6
        lifespan mice both fall with age; H3 partly via engaged weight; H4 not supported; not explained by satiety)
- [X] 8b. `007_model_free_early_trials.py` (user suggestion): easy-trial (|c| >= 0.25) accuracy and reaction time in
      the first 20/50/90 trials of each session (unbiased block, no block prior), exploration mice
      → `exploratory-analyses/007_model_free_early_trials.{py,png}`, `artifacts/007_early_trials_per_mouse.csv`
        (lifespan less accurate from the first 20 trials, AUC 0.13–0.24; falls with age within lifespan; no RT
        slowing; lifespan errors concentrated in very fast (< 80 ms) responses)
- [X] 9. **Consult user:** proposed locked tests, with adversarial review of failure scenarios
      → `exploratory-analyses/006_locked_test_proposal.md` (user approved 2026-10-12: fixed sequence
        C1a → C2a → C1b → C2b at one-sided α 0.05, plus add-on model-free test C3)

## Confirmatory analysis (only after explicit approval)
- [X] 10. `008_confirmatory_tests.py`: run locked tests on confirmation mice
      → `confirmatory-analyses/008_confirmatory_tests.{py,png}`, `008_confirmatory_results.csv`, `008_descriptives.csv`,
        `artifacts/008_*` (precondition met; C1a significant; C2a not (ρ −0.45, p 0.065), so C1b and C2b not
        evaluated under the fixed sequence; add-on C3 significant)
- [X] 10b. POST HOC (user request, not confirmatory): `confirmatory-analyses/009_posthoc_checks.py`: age overlap,
      split balance, state comparability, H3 reframed as descriptive, sessions per mouse, C3 errors by contrast
      → `confirmatory-analyses/009_posthoc_checks.{py,txt}` (age-matched lifespan still below BWM; lifespan
        engaged states much weaker; lifespan easy errors start at or before stimulus onset)
- [X] 10c. POST HOC (user request): timing/rig diagnostics and sensitivity of C1a/C3 to anticipatory trials
      → `confirmatory-analyses/011_timing_diagnostics.{py,png}`, `artifacts/011_timing_per_session.csv`,
        `confirmatory-analyses/012_sensitivity_trial_filters.{py,png}`, `012_sensitivity_results.csv`,
        `artifacts/012_metrics_*.csv`, `confirmatory-analyses/013_tail_check.py`, `artifacts/013_tail_check.csv` (early movement is a churchlandlab feature, equal in young BWM churchlandlab
        mice; occupancy gap holds under both filters; C3 gap mostly anticipatory trials); report updated
- [X] 10d. POST HOC (user request 2026-10-13, report review): C1a after excluding confirmation mice with weak engaged
      states (engaged weight < 5 or < 7)
      → `confirmatory-analyses/014_weak_state_exclusion.{py,csv}` (gap unchanged: p 4.5e-5 and 5.5e-4)
- [X] 11. **Consult user:** results; decide whether to do the neural extension
      (user: post hoc checks (10b); neural extension skipped for now; write the report)

## Optional neural extension
- [ ] 12. (skipped for now, user 2026-10-12) Load spike shards as pynapple objects; firing rate and Fano factor in engaged vs disengaged trials,
      within region and by age (needs its own exploration/confirmation plan)

## Report
- [X] 13. `report.qmd` → `report/report.html`
      → `report.qmd`, `report/report.html` (+ figure PNGs), `confirmatory-analyses/010_posthoc_figure.{py,png}`,
        `instruction-suggestions.md`
