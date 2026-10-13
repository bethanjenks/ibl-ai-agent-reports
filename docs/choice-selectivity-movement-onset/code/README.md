# Analysis code: Choice-selective neurons around movement onset

Scripts that produced the report in the folder above. Shared helpers are in `exploratory-analyses/common.py`. `000_data_census_and_split.py` creates the session-level exploration/confirmation split (seed 0). Per-session binned spike counts are cached in `artifacts/` (not included; large).

## Setup

1. Clone [int-brain-lab/ibl-ai-agent](https://github.com/int-brain-lab/ibl-ai-agent) and install its environment (the scripts import `ibl_ai_agent` for dataset lookup).
2. Get the dataset: the Brain Wide Map derived dataset `bwm_ephys` version 1.2.1 (see `docs/data_locations.md` in ibl-ai-agent).
3. List it in a `data_locations.local.yaml` (see `data_locations.example.yaml` in ibl-ai-agent) and point to that file:
   `export IBL_AGENT_DATA_LOCATIONS=/path/to/data_locations.local.yaml`

## Run

Run each script from its own folder, in numeric order: `exploratory-analyses/` first, then `confirmatory-analyses/`. Scripts write intermediate results to `artifacts/` and figures next to themselves, so later scripts depend on earlier ones. The `.csv`, `.txt` and `.log` files here are the outputs from the original run, for comparison.

`question.md`, `TODO.md` and `change-log.md` record the question, the plan and every change made along the way; the `*_risk_note.md` and `*_locked_test_proposal.md` files (where present) record metric choices and the confirmatory tests fixed before the confirmation set was touched.

The per-lab figure in the report uses anonymised lab labels; the scripts and the log print real lab names.
