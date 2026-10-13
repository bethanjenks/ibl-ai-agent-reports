# TODO: choice selectivity around movement onset

## Setup
- [X] Preflight (2026-09-07): system Python 3.9 (project needs >=3.10), uv 0.12.9
      present, no `.venv`, no Quarto, no `gh`, no `data_locations.local.yaml`,
      no `reports/datasets`, ONE cache for openalyx exists (~2 GB). 26 GB free on disk.
- [X] Preflight re-run (2026-09-07, second session): unchanged. `.venv` exists but is
      empty (only CACHEDIR.TAG); `uv run` resolves a uv-managed Python 3.13.15 but
      `ibl_ai_agent` is not importable. No `data_locations.local.yaml`, no
      `reports/datasets`, no Quarto, no `gh`. 27 GB free.
- [X] Install environment (2026-09-07). Machine is Apple Silicon but the anaconda `uv` is an
      x86_64 binary and picked an x86_64 Python, so torch (via brainbox) had no wheel. Fixed with
      `UV_CACHE_DIR=.uv-cache uv sync --python 3.12 --extra ibl --extra notebook --extra dev`.
      Use `UV_CACHE_DIR=.uv-cache uv run --no-sync python ...` for project scripts.
- [X] Downloaded `bwm_ephys` 1.2.1 and `bwm_behavior` 2.0.0 into `reports/datasets/` via
      `scripts/download_datasets.py`; `data_locations.local.yaml` written (2026-09-07).
- [X] Quarto 1.10.18 installed without sudo via `uv tool install quarto-cli` (binary at
      `~/.local/bin/quarto`; Homebrew cask failed on the sudo prompt). `gh` not installed;
      user only asked for rendering, not publishing.
- [ ] **Consult user**: approve question explication (now restricted to the initial
      unbiased block), replicate definition, data split, and how often to check in.

## Exploratory analysis (exploration set only)
- [X] 000: data census and proposed split (2026-09-07). All 459 sessions start with a
      90-trial unbiased block; after the BWM mask ~60 trials/session remain (17-87), ~25 on
      the rarer choice side (4-43). 10,059 of 75,395 good units lack a Beryl label and are
      dropped. Proposed split: 67 repeated-site sessions + 20 seeded random sessions =
      87 exploration sessions; 372 confirmation sessions. 241 Beryl regions have >= 1
      confirmation session with >= 5 units; 136 have >= 5 such sessions.
      Code: `exploratory-analyses/000_data_census_and_split.py`.
      Output: `exploratory-analyses/000_data_census_and_split.png`,
      `exploratory-analyses/000_region_coverage.csv`, `artifacts/unbiased_trials.parquet`,
      `artifacts/data_split.csv`.
- [X] 001: rasters and PSTHs by choice for 3 exploration insertions (PO/LP/VPM/VISa,
      MOp/PVT, APN/MRN/VISa/CA1), 3 units each (2 strongest by standardized L-R difference in
      [-0.1, 0.1] s, 1 random), plus per-insertion L-vs-R rate scatter (2026-09-07). The first
      run showed no pre-onset spikes: a spike-to-trial assignment bug in the shared binning
      helper, fixed and verified against a brute-force histogram. Shared helper
      `exploratory-analyses/common.py` caches 10 ms binned counts (-1..+1 s around
      firstMovement_times) per insertion in `artifacts/binned/<pid>.npz`.
      Code: `exploratory-analyses/001_rasters_psths_by_choice.py`.
      Output: `exploratory-analyses/001_rasters_psths_by_choice.png`.
- [X] 002: metric-definition figure (2026-09-07). Three per-unit scalars from per-trial rates
      in a provisional window W = [0, 0.2] s after movement onset and a 0.3 s pre-stimulus
      baseline B: delta_raw (W), delta_bl (W - B), delta_base (B); each standardized by pooled
      SD and tested by choice-label permutation (5000 shuffles). APN 344: raw d=+1.33 and
      baseline-subtracted d=+1.30 both p<1e-4, baseline d=-0.36 n.s. VISa 733: raw d=+0.73
      p=0.015 but baseline-subtracted d=+0.21 p=0.50 and baseline d=+0.72 p=0.018, i.e. its
      selectivity is entirely a whole-trial offset. Cache window widened to [-2.5, 1] s so B
      is always available; helpers `window_rates`, `baseline_rates` added to `common.py`.
      Code: `exploratory-analyses/002_metric_definition.py`.
      Output: `exploratory-analyses/002_metric_definition.png`.
- [X] 003: window grid on all 144 exploration insertions, 15,906 units, no unit selection
      (2026-09-07; 74 s runtime). Sliding 100 ms windows: fraction of units with permutation
      p < 0.05 (baseline-subtracted) sits at chance (0.04-0.05) from -0.6 to -0.2 s, rises from
      -0.15 s, peaks at 0.15 for window centres 0.05-0.10 s, decays to 0.085 by +0.5 s. Longer
      windows at start 0: 0.2 s best (0.159), 0.3 s (0.153), 0.5 s (0.137). Raw and
      baseline-subtracted fractions nearly identical; drift candidates ~1% of units; baseline-only
      fraction 0.06-0.08 (small excess of whole-trial choice-correlated offsets). Region ranking
      stable across windows: SCm 0.33, MRN 0.31, APN 0.27, VPM 0.19, PoT 0.17, VISa 0.14, PO 0.12,
      VISam 0.12, DG 0.09, LP 0.09, VISp 0.08, CA1 0.06 at W = [0, 0.2] s.
      Code: `exploratory-analyses/003_window_grid.py`.
      Output: `exploratory-analyses/003_window_grid.png`, `003_session_region_window.csv`,
      `003_window_grid.log`, `artifacts/003_unit_metrics.parquet`, `artifacts/binned/*.npz`.
- [X] 003b: wide window [-0.2, 0.2] s, mean-rate vs omnibus test (2026-09-07). User wants every
      neuron involved in the turn, preparatory or executive. Omnibus = sum over 4 x 100 ms
      sub-bins of squared baseline-subtracted standardized differences, permutation-tested.
      14,700 units: omnibus 0.175 significant vs mean-rate 0.148; 5.0% caught only by omnibus
      (sign-changing or brief selectivity), 2.2% only by mean rate. Region ranking unchanged.
      Code: `exploratory-analyses/003b_omnibus_window.py`.
      Output: `exploratory-analyses/003b_omnibus_window.png`, `artifacts/003b_wide_window_metrics.parquet`.
- [X] 004: confound checks on 15,906 exploration units (2026-09-07). Stratified permutations:
      fraction significant for choice unconditioned 0.162, choice | stimulus side 0.113,
      stimulus side | choice 0.067. Choice keeps most of its information after conditioning on
      the stimulus in every region; the reverse test is near chance except MRN (0.13). Region
      ranking preserved. Half-split: 2,575 units significant on the full block; sign of the
      strongest sub-bin agrees between block halves in 96.9%, but that figure is inflated by
      selecting on the full block. Unbiased: r(d_first, d_second) = 0.60 across all 13,516 units;
      of 1,789 units significant on the first half alone, 90.5% keep their sign in the second half
      and 35.9% are significant there too (chance 5%). Region ranking first vs second half:
      Spearman 0.49 over 25 regions. Each half alone gives 0.11 significant. Trial-index vs choice correlation within block: median |r| 0.13, 90th
      percentile 0.25. Shared statistics (`subbin_rates`, `standardized_diffs`, `omnibus_test`,
      `permuted_labels`) added to `common.py`.
      Code: `exploratory-analyses/004_confounds.py`.
      Output: `exploratory-analyses/004_confounds.png`, `artifacts/004_confound_metrics.parquet`.
- [X] 005: region-level test design on the exploration set (2026-09-07). Three tests of
      "mean per-session fraction of omnibus-significant units > chance" on 26 regions with >= 5
      exploration sessions: pooled binomial (20 regions BH q<0.05; anti-conservative by design),
      session Wilcoxon (12), session-permutation (22). On the pre-movement control window
      [-0.6, -0.2] s all three give 0 significant regions (calibrated). Session-permutation null
      mean fraction 0.043. Bootstrap power at confirmation-set session counts (Bonferroni proxy):
      ~1.0 for regions with fraction >= 0.2 and >= 12 sessions (SCm, APN, ZI, MRN, VPM, CP, MOs),
      0.7-0.99 for fraction 0.10-0.16 with >= 9 sessions (PO, LP, RN, ProS, Eth, VISp, CA3),
      < 0.5 for regions with < 7 confirmation sessions or fraction < 0.09 (DG, CA1, SUB).
      Code: `exploratory-analyses/005_region_test_design.py` (helper `omnibus_stats` added to
      `common.py`). Output: `exploratory-analyses/005_region_test_design.png`,
      `005_region_tests_exploration.csv`, `005_region_tests_control_window.csv`,
      `artifacts/005_exploration_session_region_fractions.npz`, `005_exploration_session_region_keys.csv`.
- [X] **Consult user**: plan locked 2026-09-07 (see question.md 'Locked confirmatory plan'); user
      requested BH correction for each of the three region tests separately.

## Confirmatory analysis (confirmation set, only after explicit approval)
- [X] 006: locked per-unit statistic and session x region fractions on the confirmation set
      (2026-09-07; 309 s). 368 of 372 sessions had eligible session-regions: 1,924 session-regions,
      47,917 units. Code: `confirmatory-analyses/006_confirmation_fractions.py`.
      Output: `artifacts/006_confirmation_session_region_fractions.npz`, `006_confirmation_session_region_keys.csv`,
      `confirmatory-analyses/006_confirmation_fractions.log`.
- [X] 007: locked region tests (2026-09-07). 136 Beryl regions tested (>= 5 sessions); 115
      significant by the primary session-permutation test at BH q < 0.05; 21 not. Sensitivity:
      pooled binomial agrees on 129/136 regions, session Wilcoxon on 88/136 (less powerful, as in
      exploration). Top effect sizes: SNr 0.43, NPC 0.41, RN 0.39, GPe 0.36, MRN 0.32, GRN 0.32,
      RT 0.31. Code: `confirmatory-analyses/007_region_tests.py`.
      Output: `confirmatory-analyses/007_region_tests.csv`, `007_region_tests.png`, `007_region_tests.log`.

- [X] 008: post-hoc variants on the confirmation set (2026-09-07; 73 s): pre-movement [-0.2, 0],
      post-onset [0, 0.2], choice | stimulus side, stimulus side | choice, per session-region and
      per unit. Code: `confirmatory-analyses/008_posthoc_compute.py`.
      Output: `artifacts/008_posthoc_session_region.csv`, `artifacts/008_posthoc_units.parquet`.
- [X] 009: post-hoc figures (2026-09-07). Overall session-region means: locked 0.173, pre 0.090,
      post 0.173, choice|stim 0.115, stim|choice 0.071. Post > pre in 123/136 regions; largest
      execution dominance in SSp-ll, GPe, PGRN, RN, RT. Choice|stim > stim|choice in 107/136;
      stimulus information given choice >= 0.10 mainly in motor output nuclei (SNr, GPe, GRN,
      PGRN, RN, MRN). Per-lab mean fractions 0.11-0.22, all above chance; top regions are
      reproduced across labs. Code: `confirmatory-analyses/009_posthoc_figures.py`.
      Output: `009_posthoc_windows_stimulus.png`, `009_posthoc_brain_maps.png`, `009_posthoc_per_lab.png`,
      `009_posthoc_region_summary.csv`.

## Report
- [X] Report (2026-09-07): Quarto HTML rendered to `report/index.html` (copied to `report.html`);
      figures in `report/figures/` (mouse identifiers removed from 001/002 figures; lab names
      anonymised in the per-lab figure). Instruction-file suggestions in
      `report/suggested_instruction_edits.md`. Privacy scan of the HTML: no user names, local
      paths, institution names, mouse or lab identifiers found. Publishing not done (opt-in; `gh` not installed).
