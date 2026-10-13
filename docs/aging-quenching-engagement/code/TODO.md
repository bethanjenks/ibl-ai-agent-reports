# TODO: age, engagement and FF quenching

## Planning
- [X] Read paper methods (FF definition, windows), authors' config; check posterior coverage per split
- [X] 0. **Consult user:** approve plan, split, consultation frequency
      (user 2026-10-13: reuse GLM-HMM split; all trials, first 400 as a check; check in at marked steps)

## Exploratory analysis (exploration mice only)
- [X] 1. `000_data_split.py`: copy split; count neurons per ROI × cohort, engaged / disengaged trials per session
      → `exploratory-analyses/000_data_split.{py,png}`, `quench_common.py` (shared), `artifacts/split.csv`
        (lifespan ~470 neurons, mostly MOs/VISp+pm/MBm; bwm median 12 disengaged trials per session)
- [X] 2. `001_ff_single_session.py`: one bwm + one lifespan session; rasters, condition means, residual variance,
      FR and mean-subtracted FF time courses; pre/post windows marked; engaged vs disengaged trials overlaid
      → `exploratory-analyses/001_ff_single_session.{py,png}` (bwm disengagement = end-of-session block with
        +50% FR; per-neuron quench very noisy at ~70–100 trials; FF residual variance uses n − k)
- [X] 3. `002_ff_pilot.py`: all exploration sessions; per-neuron pre/post FR, pre/post FF and quench on all / engaged / disengaged trials with
      trial-count matching; save `artifacts/002_quench_per_neuron.parquet`. Time estimate before full run.
      Adversarial checks: trial position (early vs late engaged trials), firing-rate drift, movement in the post
      window, trial-count bias of FF, θ in {0.6, 0.8, 0.9}, first-400 vs all trials
      → `exploratory-analyses/002_ff_pilot.{py,png}`, `002_metric_risk_note.md`, `artifacts/002_metrics_per_neuron.parquet`,
        `artifacts/002_trial_info.parquet` (35 s run; FF rises with trial count; placebo drift ≈ state effect;
        84% of trials move before post window ends; Part A quench opposite in sign between cohorts)
- [X] 4. **Consult user:** metric diagnostics, risk note, parameter choices
      (user 2026-10-13: Part B uses paper window plus a movement-safe variant chosen on bwm diagnostics before
      looking at age; Part A descriptive only, θ = 0.8 with 0.6 as check)
- [X] 4b. `003_movement_safe_window.py` (bwm exploration only, no age): FF time course for late-moving trials;
      choose the movement-safe post window and the trial rule (exclude trials that moved before window end)
      → `exploratory-analyses/003_movement_safe_window.{py,png}` (paper-window FF drop is mostly movement-locked;
        chosen: post [0.02, 0.12] s, first movement > 0.12 s, ~70% trials kept)
- [X] 5. `004_age_effects_exploration.py` (Part A already in step 3): Part B, age/cohort effect on each metric using engaged trials only vs all trials (matched), within
      churchlandlab and all labs (OLS with ROI fixed effects, mouse-level age permutation; equal N per neuron)
      → `exploratory-analyses/004_age_effects_exploration.{py,png}`, `artifacts/004_partB_per_neuron.parquet`,
        `artifacts/004_age_betas.csv` (β_eng ≈ β_all for every metric; age effects weak at mouse level, z ≤ 1.7)
- [X] 6. **Consult user:** proposed locked tests, adversarial review → `exploratory-analyses/004_locked_test_proposal.md`
      (user 2026-10-13: added FR and FF tests F1–F4; Δ criterion replaces ratio; approved)

## Confirmatory analysis (only after explicit approval)
- [X] 7. `confirmatory-analyses/005_confirmatory_tests.py`: locked tests on confirmation mice
      (code checked on exploration mice → `005_*_exploration_check.*`, `artifacts/005_partB_per_neuron_exploration.parquet`)
      → `confirmatory-analyses/005_confirmatory_results.csv`, `005_engaged_vs_all.csv`, `005_confirmatory_tests.png`,
        `005_confirmatory_log.txt`, `artifacts/005_partB_per_neuron_confirmation.parquet` (C1–C3 significant,
        p_holm 0.011; F3 ff_post significant, p_holm 0.02; F1, F2, F4 not; C2 Δ = −0.003 [−0.009, 0.002] vs β_all 0.029)
- [X] 8. **Consult user:** results (user 2026-10-13: add post hoc checks, write report)
- [X] 8b. POST HOC (user request): `confirmatory-analyses/006_posthoc_checks.py`: within-cohort age effects
      (confirmation mice) and per-ROI age effects (all mice, descriptive)
      → `confirmatory-analyses/006_posthoc_checks.png`, `006_posthoc_within_cohort.csv`, `006_posthoc_roi.csv`
        (within bwm: quench β 0.039 p 0.066 paper, 0.057 p 0.003 movement-safe; ff_post p 0.03; lifespan alone n.s.;
        quench β > 0 in 15/16 ROIs, CI excludes 0 in LP, LS, SCm)

## Report
- [X] 9. `report.qmd` → `report/report.html`
      → `report.qmd`, `report/report.html` (figures embedded), `instruction-suggestions.md`
- [X] 10. Report review (user request 2026-10-13): numbers checked against outputs; Discussion no longer leans on Part A;
      added engaged/disengaged trial fractions per cohort, quench scale, mouse-exclusion counts, definitions (GLM-HMM,
      sign convention, z, Holm, ROI fixed effects, presence ratio, attenuation Δ); fixed post hoc mouse counts, |z| ≤ 1.9
      → `report.qmd`, `report/report.html`
