# Question: choice selectivity around movement onset, brain-wide

## Original question (user, 2026-09-07)
Which brain regions have neurons whose firing rate around movement onset is
significantly different for left vs. right choices?

## Refinement from user (2026-09-07, second message)
Use only trials from the **initial unbiased block** (the first block of each
session, `probabilityLeft == 0.5`; nominally 90 trials before biased 0.2/0.8
blocks begin).

Consequences for the plan:
- The block-prior confound listed below is removed by design: stimulus side and
  choice are both ~50/50 and the prior is flat throughout the analysed trials.
- Trial counts per session drop to ~90 before the BWM mask, so roughly 30-45
  trials per choice. Per-unit tests will have low power. Exploration must check
  whether a unit-level selectivity fraction is still informative, or whether a
  region-level pooled statistic (e.g. session-level population decoding accuracy
  of choice, or a mixed model across units) is the better primary metric.
- All analysed trials fall in the first ~10-15 minutes of each session, so
  early-session drift/settling (caveat card `firing_rate_nonstationarity.md`)
  is the main time-in-session risk; an even/odd or first/second-half-of-block
  stability check belongs in exploration.
- Stored `features/event_response_features.parquet` cannot be used: it
  aggregates over all included trials without a choice or block split. Spike
  counts must be recomputed from the local spike shards.
- Trial selection: `probabilityLeft == 0.5` AND `bwm_include` AND `choice != 0`,
  from `bwm_ephys/metadata/trials.parquet`. Sessions whose first block is not
  0.5 (should not exist in BWM) are flagged, not silently dropped.

## Current explication (draft, not yet approved)
For each good unit in the Brain Wide Map (BWM), compare trial spike counts in a
window around `firstMovement_times` between left-choice and right-choice
trials. Call a unit *choice-selective* if the difference passes a per-unit
test. For each Beryl region, estimate the fraction of choice-selective units
and test whether it exceeds the false-positive rate expected under the null,
using sessions (or subjects) as independent replicates.

## Terms to define during exploration
- **Around movement onset**: candidate windows relative to `firstMovement_times`,
  e.g. [-0.1, +0.1], [0, +0.2], [-0.2, 0] s; choose by grid on the exploration set.
- **Choice**: `trials.choice` (-1 / +1); exclude no-go trials. Trial mask: BWM
  standard mask (reaction time, contrast, etc.) from the local trials table.
- **Significantly different**: per-unit two-sided test on trial spike counts
  (Mann-Whitney U or label-permutation); alpha and multiple-comparison control
  to be locked after exploration.
- **Confounds to handle**: choice is correlated with stimulus side on correct
  trials and with wheel movement direction by construction. Block prior is no
  longer a confound (unbiased block only). Candidate controls for stimulus
  side: condition on it, use a GLM with choice and stimulus side, or a
  stratified permutation. Decide in exploration.
- **Region**: Beryl atlas mapping of unit location; minimum units per region
  per session for eligibility.

## Replicates and data split (approved by user 2026-09-07)
- Replicate: a session's population of units within one region.
- Exploration set: repeated-site insertions present in local `bwm_ephys`
  (67 PIDs; PPC / hippocampus / thalamus) plus a small random sample of other
  sessions (seeded) so that motor-related regions are also represented.
- Confirmation set: all remaining BWM insertions, giving brain-wide coverage.

## Data surface
Local `bwm_ephys` v1.2.0 (units, trials with `probabilityLeft`, `choice`,
`firstMovement_times`, `bwm_include`; spike shards). Stored
`event_response_features` are not usable here (no choice/block split).
`bwm_behavior` is optional for this question. Not yet installed; see TODO.md.

## Candidate metrics (provisional, from step 002)
Per unit, per trial: W = firing rate in a window after movement onset (provisional [0, 0.2] s,
to be gridded), B = firing rate in the 0.3 s before stimulus onset (wheel held still by task
design). Scalars, each standardized by pooled across-trial SD and tested by choice-label
permutation:
- delta_raw = mean_L(W) - mean_R(W): candidate primary if drift is rare; inflated by whole-trial
  rate offsets that co-vary with choice runs.
- delta_bl = mean_L(W - B) - mean_R(W - B): candidate primary; removes whole-trial offsets, so
  it measures a choice difference that appears at movement.
- delta_base = mean_L(B) - mean_R(B): companion drift indicator; a unit selective in delta_raw
  and delta_base but not delta_bl is a drift candidate, not a choice cell.

## Provisional parameter choices after step 003 (not yet locked)
- Window W = [-0.2, 0.2] s around firstMovement_times (user decision 2026-09-07: include every
  neuron involved in the turn, preparatory or executive). The population time course is at
  chance before -0.2 s, rises from -0.15 s, peaks at +0.05 to +0.10 s and decays by +0.5 s.
- Primary per-unit statistic: omnibus over 4 x 100 ms sub-bins, sum_k d_k^2, where d_k is the
  baseline-subtracted standardized left-right difference in sub-bin k. Detects selectivity that
  is brief or changes sign within W, which a mean rate over W cancels (step 003b).
- Per-unit test: choice-label permutation, alpha = 0.05, 1000 shuffles.
- Post-hoc, descriptive only: split into pre-movement [-0.2, 0] and post-onset [0, 0.2] maps.

## Locked confirmatory plan (approved by user 2026-09-07)
Data
- Confirmation set: the 372 sessions not in the exploration set (`artifacts/data_split.csv`).
- Trials: initial unbiased block (`probabilityLeft == 0.5`), `bwm_include`, left or right choice,
  `firstMovement_times` present. Units: BWM good units with a Beryl label.
- Eligibility: a session x region replicate needs >= 5 units; a region is tested if it has
  >= 5 eligible confirmation sessions. Regions below that are reported as untestable, not negative.

Per-unit statistic (fixed)
- Window [-0.2, 0.2] s around `firstMovement_times`, 4 sub-bins of 100 ms.
- Per trial and sub-bin: rate minus the rate in the 0.3 s before stimulus onset.
- Omnibus statistic: sum over sub-bins of squared (mean_L - mean_R) / SD.
- 1000 choice-label permutations per session, shared by all units and probes of that session,
  seeded from the session UUID. A unit is selective if its statistic exceeds the 95th percentile
  of its own permutation null.

Per-region test (primary, one p-value per region)
- Statistic: mean over eligible sessions of the fraction of selective units.
- Null: 100,000 draws, each taking one random permutation per session independently and
  recomputing the session fraction with the same per-unit thresholds.
- One-sided p = (draws >= observed + 1) / (draws + 1). Benjamini-Hochberg across tested regions,
  q < 0.05 defines the answer set. Effect size: mean fraction with bootstrap 95% CI over sessions.

Sensitivity (reported, not used for claims): pooled binomial and one-sided session Wilcoxon, each
BH-corrected separately across the same tested regions so all three are on the same footing.
Post-hoc descriptive only: pre-movement [-0.2, 0] vs post-onset [0, 0.2] fractions; choice
conditioned on stimulus side; brain-map figures; per-lab breakdown for the top regions.

Adversarial notes
- Power is ~1 for regions with fraction >= 0.2 and >= 12 sessions, 0.7-0.99 for fraction 0.10-0.16
  with >= 9 sessions, and < 0.5 for fraction < 0.09 or < 7 sessions. Hippocampal regions may
  therefore come out negative even if a small selective population exists; the report will say
  "not detected", not "absent".
- Exploration sessions are repeated-site heavy; region composition differs in confirmation, but
  the test is within-region so this only affects which regions are testable.
- All three region tests were calibrated on a pre-movement control window (0 or 1 of 26 regions
  significant, as expected at alpha 0.05).
