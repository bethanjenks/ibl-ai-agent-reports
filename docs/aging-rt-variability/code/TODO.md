# TODO: age and reaction time

## Planning
- [X] Inspect package docs and tables; draft question.md
- [X] 0. **Consult user:** approve plan, split strategy and consultation frequency
      (user: 30/70 split; response time as primary RT; check in at each marked step)
- [X] 1. `000_determine_data_split.py`: split mice into exploration/confirmation, stratified by cohort and lab
      (and age within bwm); save `artifacts/split.csv`; plot age/lab balance
      → `exploratory-analyses/000_determine_data_split.{py,png}`, `artifacts/split.csv`
        (exploration 42 bwm + 6 lifespan mice; confirmation 97 + 13)

## Exploratory analysis (exploration mice only)
- [X] 2. `001_rt_definitions.py`: build per-trial RT table (response time primary, movement-onset RT secondary) for exploration sessions; save
      to `artifacts/`; metric-validation figure: example session event timeline, RT distributions per contrast,
      early-movement (firstMovement < stimOn) trials, mass removed by the 0.08–2 s window
      → `exploratory-analyses/001_rt_definitions.{py,png}`, `001_metric_risk_note.md`,
        `artifacts/001_trials_rt_exploration.parquet`
- [X] 3. **Consult user:** RT definition, trial inclusion, and window choice
      (user: paper window primary; grid over window, early-movement handling and variability measure; add tail rates)
- [X] 4. `002_variability_metrics.py`: didactic figure for median RT and CV / robust CV per session; check
      mean–SD coupling; per-contrast vs pooled; session-to-session reliability within mouse
      → `exploratory-analyses/002_variability_metrics.{py,png}`, `rt_metrics.py` (shared metric code),
        `artifacts/002_session_metrics_primary.csv`
- [X] 5. `003_parameter_grid.py`: effect of age across RT window, trial count (first 400 vs all), CV variant;
      adversarial checks: cohort-only effect, lab, time in session, trial number, contrast mix, session order
      → `exploratory-analyses/003_parameter_grid.{py,png}`, `artifacts/003_grid_age_rho.csv`, `003_tail_age_rho.csv`,
        `003_session_metrics_grid.parquet` (grid done here; covariate/lab/cohort checks moved to step 6)
- [X] 6. `004_age_effects_exploration.py`: per-mouse metric vs age plots; mixed models; within-cohort and
      overlap-window comparisons
      → `exploratory-analyses/004_age_effects_exploration.{py,png}`, `004_locked_test_proposal.md`,
        `artifacts/004_mixed_models_exploration.csv`
- [X] 7. **Consult user:** proposed locked tests, with adversarial review of failure scenarios
      (user approved P1, S1–S4 as written in `exploratory-analyses/004_locked_test_proposal.md`)

## Confirmatory analysis (only after explicit approval)
- [X] 8. Run locked tests on confirmation mice
      → `confirmatory-analyses/005_confirmatory_tests.{py,png}`, `005_confirmatory_results.csv`,
        `artifacts/005_trials_rt_confirmation.parquet`, `005_session_metrics_confirmation.csv`, `005_mouse_metrics_confirmation.csv`
        (P1, S1, S2, S3 significant; S4 not)
- [X] 9. **Consult user:** results (user asked for the report)
- [X] 9b. `006_posthoc_descriptives.py`: pre-declared descriptives on confirmation set (grid sensitivity of
      sd_log_rt and CV; median RT across both cohorts; tail rates); no tests
      → `confirmatory-analyses/006_posthoc_descriptives.{py,png}`, `006_window_sensitivity.csv`
        (sd_log_rt effect shrinks with wider windows in confirmation: ρ 0.24 at 5 s, 0.00 uncut; RT < 0.15 s rate rises with age)

## Report
- [X] 10. `report.qmd` → `report/report.html`
      → `report.qmd`, `report/report.html` (+ figure PNGs for lightbox), `instruction-suggestions.md`
- [X] 11. Report review (user request): correctness and readability fixes
      → `confirmatory-analyses/007_review_checks.{py,csv}` (lower RT bound 0.15/0.2 s; paper session QC; S4 bootstrap CI);
        plotting-only fixes to 002–005 figure scripts (saved results unchanged, md5-checked); `report.qmd` text revised;
        `report/report.html` re-rendered
