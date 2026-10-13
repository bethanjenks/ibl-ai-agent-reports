"""Confound checks on the exploration set: stimulus side, and within-block stability.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

(a) Choice agrees with stimulus side on ~87% of trials. Stratified permutations shuffle choice
    only within trials of the same stimulus side (left / right / zero contrast), testing choice
    information beyond the stimulus; the mirror test shuffles stimulus side within choice.
(b) Drift: the block is split into first and second halves; the sign of the strongest sub-bin
    difference should agree between halves for genuine choice cells, and per-session
    correlation between trial index and choice shows how much room drift has to leak in.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import (load_tables, binned_counts, omnibus_test, permuted_labels, PROJECT, SUB_EDGES)

N_PERM = 1000
ALPHA = 0.05
MIN_UNITS = 5
rng = np.random.default_rng(2)

units, trials, split = load_tables()
pids = sorted(units.loc[units.eid.isin(split.eid[split.set == "exploration"]), "pid"].unique())
rows, sess = [], []
for pid in pids:
    counts, u, t = binned_counts(pid, units, trials)
    left = (t.side == "left").to_numpy(); stim_left = (t.stim_side == "left").to_numpy()
    stim = t.stim_side.to_numpy(); half = (np.arange(len(t)) >= len(t) / 2)
    r = {"pid": pid, "eid": u.eid, "cluster_id": u.cluster_id, "region": u.beryl_acronym}
    # unconditioned choice, choice | stimulus, stimulus | choice
    _, r["p_choice"], d_sub = omnibus_test(counts, t, permuted_labels(left, rng, N_PERM))
    _, r["p_choice_given_stim"], _ = omnibus_test(counts, t, permuted_labels(left, rng, N_PERM, strata=stim))
    _, r["p_stim_given_choice"], _ = omnibus_test(counts, t, permuted_labels(stim_left, rng, N_PERM, strata=left))
    # half-split: sign of the strongest sub-bin (chosen on the full block) in each half
    k_max = np.nanargmax(np.abs(np.nan_to_num(d_sub)), axis=0)
    for name, m in (("first", ~half), ("second", half)):
        tm = t[m].reset_index(drop=True)
        _, p_h, d_h = omnibus_test(counts[:, m], tm, permuted_labels(left[m], rng, N_PERM))
        r[f"p_{name}"] = p_h; r[f"d_{name}"] = d_h[k_max, np.arange(len(u))]
    rows.append(pd.DataFrame(r))
    sess.append({"eid": u.eid[0], "n_trials": len(t), "agreement": (left == stim_left)[stim != "zero"].mean(),
                 "r_trial_choice": np.corrcoef(np.arange(len(t)), left)[0, 1],
                 "n_left_first": left[~half].sum(), "n_left_second": left[half].sum()})
um = pd.concat(rows, ignore_index=True).dropna(subset=["p_choice"]); sess = pd.DataFrame(sess)
um.to_parquet(PROJECT / "artifacts/004_confound_metrics.parquet")

sig = lambda c: (um[c] < ALPHA)
sr = um.assign(s_choice=sig("p_choice"), s_cgs=sig("p_choice_given_stim"), s_sgc=sig("p_stim_given_choice")) \
       .groupby(["eid", "region"]).agg(n=("cluster_id", "size"), s_choice=("s_choice", "mean"),
                                       s_cgs=("s_cgs", "mean"), s_sgc=("s_sgc", "mean")).reset_index()
sr = sr[sr.n >= MIN_UNITS]
reg = sr.groupby("region").agg(n_sess=("eid", "nunique"), choice=("s_choice", "mean"),
                               choice_given_stim=("s_cgs", "mean"), stim_given_choice=("s_sgc", "mean"))
reg = reg[reg.n_sess >= 5].sort_values("choice", ascending=False)
print(f"units {len(um)}; fraction significant: choice {sig('p_choice').mean():.3f}, "
      f"choice|stim {sig('p_choice_given_stim').mean():.3f}, stim|choice {sig('p_stim_given_choice').mean():.3f}")
print(reg.round(3).to_string())
cells = um[sig("p_choice")]
agree = np.sign(cells.d_first) == np.sign(cells.d_second)
print(f"half-split: units significant on full block {len(cells)}; sign agreement of strongest sub-bin "
      f"{agree.mean():.3f} (chance 0.5); significant in first half {sig('p_first').mean():.3f}, second half {sig('p_second').mean():.3f}")
print(f"sessions: median |r(trial index, choice)| {sess.r_trial_choice.abs().median():.2f}, "
      f"90th pct {sess.r_trial_choice.abs().quantile(0.9):.2f}; median choice-stimulus agreement {sess.agreement.median():.2f}")

fig, axes = plt.subplots(1, 4, figsize=(25, 6))
ax = axes[0]
y = np.arange(len(reg))
for k, (c, lab) in enumerate((("choice", "choice (unconditioned)"), ("choice_given_stim", "choice | stimulus side"),
                              ("stim_given_choice", "stimulus side | choice"))):
    ax.barh(y + (k - 1) * 0.27, reg[c], 0.27, label=lab)
ax.set_yticks(y); ax.set_yticklabels([f"{r} ({n})" for r, n in zip(reg.index, reg.n_sess)], fontsize=8); ax.invert_yaxis()
ax.axvline(ALPHA, c="k", ls="--", lw=0.8); ax.set_xlabel("fraction of units p < 0.05"); ax.legend(fontsize=8)
ax.set_title("Stratified permutation tests, per region (n sessions)")
ax = axes[1]
ax.scatter(reg.choice_given_stim, reg.stim_given_choice, s=25)
for r, row in reg.iterrows(): ax.annotate(r, (row.choice_given_stim, row.stim_given_choice), fontsize=8, xytext=(3, 3), textcoords="offset points")
lim = reg[["choice_given_stim", "stim_given_choice"]].max().max() * 1.15
ax.plot([0, lim], [0, lim], "k--", lw=0.7); ax.axhline(ALPHA, c="grey", lw=0.5); ax.axvline(ALPHA, c="grey", lw=0.5)
ax.set_xlabel("fraction significant: choice | stimulus"); ax.set_ylabel("fraction significant: stimulus | choice")
ax.set_title("Which variable carries the unique information?")
ax = axes[2]
ax.scatter(um.d_first, um.d_second, s=4, alpha=0.2, label="all units")
ax.scatter(cells.d_first, cells.d_second, s=6, alpha=0.6, label=f"significant on full block ({agree.mean():.0%} same sign)")
ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("d in strongest sub-bin, first half of block"); ax.set_ylabel("same, second half"); ax.legend(fontsize=8)
ax.set_title("Half-split stability of selectivity")
ax = axes[3]
ax.hist(sess.r_trial_choice, 25); ax.set_xlabel("correlation(trial index, choice) within block"); ax.set_ylabel("sessions")
ax.set_title(f"Room for drift to leak in\nmedian |r| = {sess.r_trial_choice.abs().median():.2f}")
fig.tight_layout(); fig.savefig(PROJECT / "exploratory-analyses/004_confounds.png", dpi=110)
