"""Post-hoc figures on the confirmation set (descriptive, pre-declared in the locked plan).

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

A: pre-movement vs post-onset fraction per region. B: choice | stimulus vs stimulus | choice.
C: Swanson flatmaps of the locked fraction and the post-minus-pre difference.
D: per-lab breakdown of session fractions for the densest regions and overall.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iblatlas.atlas import BrainRegions
from iblatlas.plots import plot_swanson_vector
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "exploratory-analyses"))
from common import PROJECT, ARTIFACTS

res = pd.read_csv(PROJECT / "confirmatory-analyses/007_region_tests.csv")
sr = pd.read_csv(ARTIFACTS / "008_posthoc_session_region.csv")
sr = sr[sr.region.isin(res.region)]
reg = sr.groupby("region")[[c for c in sr.columns if c.startswith("frac_")]].mean().join(res.set_index("region")[["n_sessions", "significant", "mean_frac"]])
reg = reg.sort_values("mean_frac", ascending=False)
reg.to_csv(PROJECT / "confirmatory-analyses/009_posthoc_region_summary.csv")
print(f"{len(reg)} regions. Overall session-region means: " + ", ".join(f"{c[5:]} {sr[c].mean():.3f}" for c in sr.columns if c.startswith("frac_")))
print("pre > post in", int((reg.frac_pre > reg.frac_post).sum()), "regions; top 10 by pre-movement fraction:",
      reg.sort_values("frac_pre", ascending=False).head(10).index.tolist())
print("largest post - pre:", reg.assign(d=reg.frac_post - reg.frac_pre).sort_values("d", ascending=False).head(8).d.round(3).to_dict())
print("choice|stim > stim|choice in", int((reg.frac_cgs > reg.frac_sgc).sum()), "regions; regions where stim|choice >= 0.10:",
      reg[reg.frac_sgc >= 0.10].frac_sgc.round(3).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# A: pre vs post
ax = axes[0]
c = np.where(reg.significant, "C0", "grey")
ax.scatter(reg.frac_pre, reg.frac_post, s=np.sqrt(reg.n_sessions) * 8, c=c, alpha=0.7)
for r, row in reg.head(15).iterrows(): ax.annotate(r, (row.frac_pre, row.frac_post), fontsize=8, xytext=(3, 3), textcoords="offset points")
lim = max(reg.frac_pre.max(), reg.frac_post.max()) * 1.1; ax.plot([0, lim], [0, lim], "k--", lw=0.7)
ax.set_xlabel("fraction selective, pre-movement [-0.2, 0] s"); ax.set_ylabel("fraction selective, post-onset [0, 0.2] s")
ax.set_title("A. Preparation vs execution (marker size ~ sessions; grey = not significant in locked test)")
# B: choice|stim vs stim|choice
ax = axes[1]
ax.scatter(reg.frac_cgs, reg.frac_sgc, s=np.sqrt(reg.n_sessions) * 8, c=c, alpha=0.7)
for r, row in reg.sort_values("frac_cgs", ascending=False).head(12).iterrows(): ax.annotate(r, (row.frac_cgs, row.frac_sgc), fontsize=8, xytext=(3, 3), textcoords="offset points")
for r, row in reg.sort_values("frac_sgc", ascending=False).head(5).iterrows(): ax.annotate(r, (row.frac_cgs, row.frac_sgc), fontsize=8, xytext=(3, -9), textcoords="offset points", color="C3")
lim = max(reg.frac_cgs.max(), reg.frac_sgc.max()) * 1.1; ax.plot([0, lim], [0, lim], "k--", lw=0.7)
ax.axhline(0.05, c="grey", lw=0.5); ax.axvline(0.05, c="grey", lw=0.5)
ax.set_xlabel("fraction selective for choice | stimulus side"); ax.set_ylabel("fraction selective for stimulus side | choice")
ax.set_title("B. Stratified permutations: unique choice vs unique stimulus information")
fig.tight_layout(); fig.savefig(PROJECT / "confirmatory-analyses/009_posthoc_windows_stimulus.png", dpi=110)

# C: Swanson flatmaps
br = BrainRegions()
fig, axes = plt.subplots(2, 1, figsize=(16, 13))
ax = axes[0]
plot_swanson_vector(reg.index.to_numpy(), reg.mean_frac.to_numpy(), ax=ax, br=br, cmap="viridis", vmin=0, vmax=0.4,
                    annotate=True, annotate_n=12, annotate_order="top", show_cbar=True, fontsize=7, empty_color="w")
ax.set_title("C1. Fraction of choice-selective units, locked test (136 tested regions; white = untested)"); ax.axis("off")
ax = axes[1]
d = (reg.frac_post - reg.frac_pre).to_numpy()
plot_swanson_vector(reg.index.to_numpy(), d, ax=ax, br=br, cmap="RdBu_r", vmin=-0.2, vmax=0.2, show_cbar=True, empty_color="w",
                    annotate=True, annotate_n=8, annotate_order="top", fontsize=7)
ax.set_title("C2. Post-onset minus pre-movement fraction (red = execution-dominated)"); ax.axis("off")
fig.tight_layout(); fig.savefig(PROJECT / "confirmatory-analyses/009_posthoc_brain_maps.png", dpi=110)

# D: per-lab breakdown
top = reg.head(10).index.tolist()
labs = sorted(sr.lab.unique()); colors = dict(zip(labs, plt.cm.tab10(np.linspace(0, 1, len(labs)))))
fig, axes = plt.subplots(1, 2, figsize=(20, 6.5), gridspec_kw={"width_ratios": [3, 1]})
ax = axes[0]
for i, r in enumerate(top):
    g = sr[sr.region == r]
    for lab, gg in g.groupby("lab"):
        ax.scatter(np.full(len(gg), i) + np.random.default_rng(0).uniform(-0.25, 0.25, len(gg)), gg.frac_locked, s=18, color=colors[lab], label=lab)
    ax.hlines(g.frac_locked.mean(), i - 0.35, i + 0.35, color="k", lw=2)
h, l = ax.get_legend_handles_labels(); uniq = dict(zip(l, h)); ax.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=2, title="lab")
ax.set_xticks(range(len(top))); ax.set_xticklabels([f"{r}\n({int(reg.n_sessions[r])} sess.)" for r in top], fontsize=8)
ax.axhline(0.05, c="grey", ls="--", lw=0.8); ax.set_ylabel("fraction selective per session (locked test)"); ax.set_title("D1. Densest regions: one dot per session, coloured by lab")
ax = axes[1]
lab_mean = sr.groupby("lab").agg(f=("frac_locked", "mean"), n=("eid", "nunique")).sort_values("f")
ax.barh(lab_mean.index, lab_mean.f, color=[colors[l] for l in lab_mean.index])
for i, (l, row) in enumerate(lab_mean.iterrows()): ax.text(row.f + 0.003, i, f"n={row.n}", va="center", fontsize=8)
ax.axvline(sr.frac_locked.mean(), c="k", ls="--", lw=0.8); ax.set_xlabel("mean session-region fraction (all regions)"); ax.set_title("D2. Per lab (n sessions)")
fig.tight_layout(); fig.savefig(PROJECT / "confirmatory-analyses/009_posthoc_per_lab.png", dpi=110)
print("per-lab mean fraction:", lab_mean.f.round(3).to_dict())
