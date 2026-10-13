# Analysis code: Reaction-time variability increases with age

Scripts that produced the report in the folder above. `artifacts/split.csv` is the locked mouse-level exploration/confirmation split; `000_determine_data_split.py` regenerates it identically (seed 20261008).

## Setup

1. Clone [int-brain-lab/ibl-ai-agent](https://github.com/int-brain-lab/ibl-ai-agent) and install its environment (the scripts import `ibl_ai_agent` for dataset lookup).
2. Get the dataset: `ibl_aging_zang2025` version 1.0.0. This is a local package built from the public OpenAlyx release tagged `2025_Q3_Zang_et_al_Aging` (all 497 sessions) and is not published separately: ask the author for a copy, or rebuild it with the ibl-ai-agent `data-ingest` workflow.
3. List it in a `data_locations.local.yaml` (see `data_locations.example.yaml` in ibl-ai-agent) and point to that file:
   `export IBL_AGENT_DATA_LOCATIONS=/path/to/data_locations.local.yaml`

## Run

Run each script from its own folder, in numeric order: `exploratory-analyses/` first, then `confirmatory-analyses/`. Scripts write intermediate results to `artifacts/` and figures next to themselves, so later scripts depend on earlier ones. The `.csv`, `.txt` and `.log` files here are the outputs from the original run, for comparison.

`question.md`, `TODO.md` and `change-log.md` record the question, the plan and every change made along the way; the `*_risk_note.md` and `*_locked_test_proposal.md` files (where present) record metric choices and the confirmatory tests fixed before the confirmation set was touched.
