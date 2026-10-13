"""Wide window [-0.2, 0.2] s: mean-rate difference vs an omnibus statistic over 100 ms sub-bins.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

The user wants every neuron involved in the turn, preparatory or executive. A mean rate over a
wide window cancels units whose selectivity changes sign in time. The omnibus statistic
sum_k d_k^2 over the K = 4 sub-bins (baseline-subtracted standardized differences) detects a
choice difference anywhere in the window, in either direction. Both are permutation-tested.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_tables, binned_counts, baseline_rates, T_EDGES, PROJECT

WIDE = (-0.2, 0.2)
SUB_EDGES = np.round(np.arange(WIDE[0], WIDE[1] + 0.05, 0.1), 2)     # 4 x 100 ms
N_PERM = 1000
ALPHA = 0.05
MIN_UNITS = 5

def rates(counts, edges):
    """Per-unit, per-trial mean rate in each [edges[k], edges[k+1]) -> (K, n_units, n_trials)."""
    csum = np.concatenate([np.zeros_like(counts[:, :, :1]), np.cumsum(counts, axis=2)], axis=2)
    i = np.searchsorted(T_EDGES, edges)
    return (csum[:, :, i[1:]] - csum[:, :, i[:-1]]).transpose(2, 0, 1) / np.diff(edges)[:, None, None]

units, trials, split = load_tables()
rng = np.random.default_rng(1)
pids = sorted(units.loc[units.eid.isin(split.eid[split.set == "exploration"]), "pid"].unique())
rows = []
for pid in pids:
    counts, u, t = binned_counts(pid, units, trials)
    left = (t.side == "left").to_numpy()
    labels = np.vstack([left, [rng.permutation(left) for _ in range(N_PERM)]]).astype(float)   # (p+1, t)
    base = baseline_rates(counts, t)
    sub = rates(counts, SUB_EDGES) - base[None]                    # (K, u, t)
    wide = rates(counts, np.array(WIDE))[0] - base                 # (u, t)
    def d(x):   # standardized L-R difference for every permutation -> (..., p+1)
        return (x @ labels.T / left.sum() - x @ (1 - labels).T / (~left).sum()) / x.std(-1, keepdims=True)
    d_wide = d(wide)                                               # (u, p+1)
    omni = np.nansum(d(sub) ** 2, axis=0)                          # (u, p+1); silent sub-bins contribute 0
    rows.append(pd.DataFrame({"pid": pid, "eid": u.eid, "cluster_id": u.cluster_id, "region": u.beryl_acronym,
        "d_wide": d_wide[:, 0], "p_wide": (np.abs(d_wide[:, 1:]) >= np.abs(d_wide[:, :1])).mean(1),
        "omni": omni[:, 0], "p_omni": (omni[:, 1:] >= omni[:, :1]).mean(1)}))
um = pd.concat(rows, ignore_index=True).dropna(subset=["d_wide"])
um.to_parquet(PROJECT / "artifacts/003b_wide_window_metrics.parquet")

sr = um.groupby(["eid", "region"]).agg(n=("cluster_id", "size"), frac_wide=("p_wide", lambda p: (p < ALPHA).mean()),
                                       frac_omni=("p_omni", lambda p: (p < ALPHA).mean())).reset_index()
sr = sr[sr.n >= MIN_UNITS]
reg = sr.groupby("region").agg(n_sess=("eid", "nunique"), frac_wide=("frac_wide", "mean"), frac_omni=("frac_omni", "mean"))
reg = reg[reg.n_sess >= 5].sort_values("frac_omni", ascending=False)
both = ((um.p_wide < ALPHA) & (um.p_omni < ALPHA)).mean(); only_w = ((um.p_wide < ALPHA) & (um.p_omni >= ALPHA)).mean()
only_o = ((um.p_wide >= ALPHA) & (um.p_omni < ALPHA)).mean()
print(f"units: {len(um)}; significant by both {both:.3f}, mean-rate only {only_w:.3f}, omnibus only {only_o:.3f}")
print(f"overall fraction significant: mean-rate {(um.p_wide < ALPHA).mean():.3f}, omnibus {(um.p_omni < ALPHA).mean():.3f}")
print(reg.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
ax = axes[0]; ax.scatter(reg.frac_wide, reg.frac_omni, s=25)
for r, row in reg.iterrows(): ax.annotate(r, (row.frac_wide, row.frac_omni), fontsize=8, xytext=(3, 3), textcoords="offset points")
lim = reg[["frac_wide", "frac_omni"]].max().max() * 1.1; ax.plot([0, lim], [0, lim], "k--", lw=0.7)
ax.set_xlabel("fraction significant, mean rate over [-0.2, 0.2] s"); ax.set_ylabel("fraction significant, omnibus (4 x 100 ms)")
ax.set_title("Per region (session-averaged)")
ax = axes[1]
ex = um[(um.p_omni < ALPHA) & (um.p_wide >= ALPHA)].sort_values("omni", ascending=False).head(6)
pid_cache = {}
for k, (_, row) in enumerate(ex.iterrows()):
    if row.pid not in pid_cache: pid_cache[row.pid] = binned_counts(row.pid, units, trials)
    counts, u, t = pid_cache[row.pid]; iu = int(np.where(u.cluster_id == row.cluster_id)[0][0]); left = (t.side == "left").to_numpy()
    sub = rates(counts, SUB_EDGES)[:, iu] - baseline_rates(counts, t)[iu]                    # (K, t)
    dd = (sub[:, left].mean(1) - sub[:, ~left].mean(1)) / sub.std(1)
    ax.plot(SUB_EDGES[:-1] + 0.05, dd, marker="o", label=f"{row.region} {row.cluster_id}")
ax.axhline(0, c="k", lw=0.7); ax.axvline(0, c="k", lw=0.7); ax.set_xlabel("sub-bin centre (s)"); ax.set_ylabel("standardized L-R difference per sub-bin")
ax.set_title("Units caught only by the omnibus test:\nselectivity that changes sign or is brief"); ax.legend(fontsize=7)
ax = axes[2]
ax.hist(um.omni[um.p_omni >= ALPHA], np.linspace(0, 6, 60), alpha=0.6, label="not significant")
ax.hist(um.omni[um.p_omni < ALPHA], np.linspace(0, 6, 60), alpha=0.6, label="p < 0.05")
ax.set_xlabel("omnibus statistic sum d_k^2"); ax.set_ylabel("units"); ax.set_yscale("log"); ax.legend(); ax.set_title("Omnibus statistic across all units")
fig.tight_layout(); fig.savefig(PROJECT / "exploratory-analyses/003b_omnibus_window.png", dpi=110)
