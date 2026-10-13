"""Data census and exploration/confirmation split.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

This step (a) builds the table of analysable unbiased-block trials, (b) counts how many
trials per choice each session offers, (c) maps good units to Beryl regions, and
(d) proposes a session-level split: repeated-site sessions plus a seeded random sample of
other sessions for exploration, everything else for confirmation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATASET_DIR = resolve_dataset_dir("bwm_ephys")          # bwm_ephys 1.2.1
PROJECT = Path(__file__).resolve().parents[1]
REPEATED_SITE_CSV = PROJECT.parents[1] / "skills/ibl-load/references/repeated_site_pids.csv"
N_EXTRA_EXPLORATION_SESSIONS = 20   # random non-repeated-site sessions added for brain-wide exploration
SEED = 0
MIN_UNITS_PER_REGION = 5            # provisional eligibility for a session x region replicate

# ---- load
units = pd.read_parquet(DATASET_DIR / "metadata/units.parquet")
trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
insertions = pd.read_parquet(DATASET_DIR / "metadata/insertions.parquet", columns=["pid", "eid", "subject", "lab"])
repeated = pd.read_csv(REPEATED_SITE_CSV, dtype=str)

# ---- analysable trials: initial unbiased block (probabilityLeft == 0.5; always the first 90
# trials in BWM sessions), BWM inclusion mask, a left or right choice, and a movement onset.
ub = trials[(trials.probabilityLeft == 0.5) & trials.bwm_include & (trials.choice != 0)
            & trials.firstMovement_times.notna()].copy()
ub["side"] = np.where(ub.choice == -1, "right", "left")   # IBL convention: choice -1 = CW wheel = right choice
ub["stim_side"] = np.where(ub.contrastRight.notna() & (ub.contrastRight > 0), "right",
                  np.where(ub.contrastLeft.notna() & (ub.contrastLeft > 0), "left", "zero"))
ub.to_parquet(PROJECT / "artifacts/unbiased_trials.parquet")

per_session = ub.groupby("eid").side.value_counts().unstack(fill_value=0)
per_session["total"] = per_session.sum(axis=1)
per_session["min_side"] = per_session[["left", "right"]].min(axis=1)

# ---- split at the session level so every insertion of a session lands in one set
rs_eids = set(insertions.loc[insertions.pid.isin(repeated.pid), "eid"])
other_eids = np.array(sorted(set(insertions.eid) - rs_eids))
extra = set(np.random.default_rng(SEED).choice(other_eids, N_EXTRA_EXPLORATION_SESSIONS, replace=False))
split = pd.DataFrame({"eid": sorted(set(insertions.eid))})
split["set"] = np.where(split.eid.isin(rs_eids | extra), "exploration", "confirmation")
split["repeated_site"] = split.eid.isin(rs_eids)
split.to_csv(PROJECT / "artifacts/data_split.csv", index=False)

# ---- region coverage (Beryl). Units without a Beryl label have only coarse parent labels
# (e.g. 'MB', 'arb', 'STR') and are dropped.
n_unlabelled = units.beryl_acronym.isna().sum()
u = units.dropna(subset=["beryl_acronym"]).merge(split, on="eid")
u = u.merge(per_session[["min_side"]], left_on="eid", right_index=True)
sess_region = u.groupby(["set", "beryl_acronym", "eid"]).agg(n_units=("cluster_id", "size"),
                                                             subject=("subject", "first"),
                                                             min_side=("min_side", "first")).reset_index()
eligible = sess_region[sess_region.n_units >= MIN_UNITS_PER_REGION]
coverage = eligible.groupby(["beryl_acronym", "set"]).agg(n_sessions=("eid", "nunique"),
                                                          n_subjects=("subject", "nunique"),
                                                          n_units=("n_units", "sum")).unstack("set", fill_value=0)
coverage.columns = [f"{a}_{b}" for a, b in coverage.columns]
coverage = coverage.sort_values("n_sessions_confirmation", ascending=False)
coverage.to_csv(PROJECT / "exploratory-analyses/000_region_coverage.csv")

# ---- summary
print(f"sessions {trials.eid.nunique()}, insertions {len(insertions)}, good units {len(units)} "
      f"({n_unlabelled} without Beryl label dropped)")
print("analysable unbiased-block trials per session:\n", per_session.total.describe().round(1).to_string())
print("trials on the rarer choice side per session:\n", per_session.min_side.describe().round(1).to_string())
print("stimulus side among analysable trials:", ub.stim_side.value_counts(normalize=True).round(3).to_dict())
print("choice-vs-stimulus agreement (non-zero contrast):",
      (ub.loc[ub.stim_side != "zero", "side"] == ub.loc[ub.stim_side != "zero", "stim_side"]).mean().round(3))
print(split.set.value_counts().to_dict(), "| repeated-site sessions:", len(rs_eids))
print(f"Beryl regions with >= 1 eligible session (>= {MIN_UNITS_PER_REGION} units): "
      f"exploration {(coverage.n_sessions_exploration > 0).sum()}, confirmation {(coverage.n_sessions_confirmation > 0).sum()}, "
      f"confirmation with >= 5 sessions: {(coverage.n_sessions_confirmation >= 5).sum()}")
print(coverage.head(15).to_string())

# ---- figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.hist([per_session.left, per_session.right], bins=np.arange(0, 60, 3), label=["left choice", "right choice"])
ax.set_xlabel("analysable unbiased-block trials per session"); ax.set_ylabel("sessions"); ax.legend()
ax.set_title("Trials per choice after BWM mask\n(unbiased block = first 90 trials)")
ax = axes[1]
ax.scatter(per_session.left, per_session.right, s=8, alpha=0.5)
ax.plot([0, 60], [0, 60], "k--", lw=0.8); ax.set_xlabel("left-choice trials"); ax.set_ylabel("right-choice trials")
ax.set_title("Left/right balance per session")
ax = axes[2]
top = coverage.head(40)
y = np.arange(len(top))
ax.barh(y - 0.2, top.n_sessions_confirmation, 0.4, label="confirmation")
ax.barh(y + 0.2, top.n_sessions_exploration, 0.4, label="exploration")
ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=7); ax.invert_yaxis()
ax.set_xlabel(f"sessions with >= {MIN_UNITS_PER_REGION} good units"); ax.legend()
ax.set_title("Region coverage (top 40 Beryl regions)")
fig.tight_layout()
fig.savefig(PROJECT / "exploratory-analyses/000_data_census_and_split.png", dpi=120)
