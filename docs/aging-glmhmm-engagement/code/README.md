# Analysis code: Older mice spend less time engaged in the IBL decision task

> **This analysis builds on the [RT-variability report](https://bethanjenks.github.io/ibl-ai-agent-reports/aging-rt-variability/).** It reuses that report's mouse split and RT metrics, so those scripts must be run first. See [Dependency on the RT-variability report](#dependency-on-the-rt-variability-report).

Scripts that produced the report in the folder above. `artifacts/split.csv` is the locked mouse-level exploration/confirmation split, shared with the RT-variability report.

## Setup

1. Clone [int-brain-lab/ibl-ai-agent](https://github.com/int-brain-lab/ibl-ai-agent) and install its environment (the scripts import `ibl_ai_agent` for dataset lookup). The GLM-HMM fits also need `nemos` (0.2.9) and `jax` (0.11.2), which are not in its lockfile.
2. Get the dataset: `ibl_aging_zang2025` version 1.0.0. This is a local package built from the public OpenAlyx release tagged `2025_Q3_Zang_et_al_Aging` (all 497 sessions) and is not published separately: ask the author for a copy, or rebuild it with the ibl-ai-agent `data-ingest` workflow.
3. List it in a `data_locations.local.yaml` (see `data_locations.example.yaml` in ibl-ai-agent) and point to that file:
   `export IBL_AGENT_DATA_LOCATIONS=/path/to/data_locations.local.yaml`

## Run

Run each script from its own folder, in numeric order: `exploratory-analyses/` first, then `confirmatory-analyses/`. Scripts write intermediate results to `artifacts/` and figures next to themselves, so later scripts depend on earlier ones. The `.csv`, `.txt` and `.log` files here are the outputs from the original run, for comparison.

`question.md`, `TODO.md` and `change-log.md` record the question, the plan and every change made along the way; the `*_risk_note.md` and `*_locked_test_proposal.md` files (where present) record metric choices and the confirmatory tests fixed before the confirmation set was touched.

## Dependency on the RT-variability report

`000_data_split.py` copies the split from `../../aging-rt-variability/code/artifacts/split.csv`, and `006_age_effects_exploration.py` reads `002_session_metrics_primary.csv` from that folder. Run the RT-variability scripts up to `002_variability_metrics.py` first.
