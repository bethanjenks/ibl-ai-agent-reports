"""Window grid on the exploration set: where in time, and with what window, is choice selectivity found?

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

For every good unit of every exploration insertion and every candidate window W we compute
delta_raw (W), delta_bl (W - pre-stimulus baseline B) and delta_base (B), standardized by the
pooled across-trial SD, with p-values from N_PERM choice-label permutations shared by all units
of a session. Results are cached per unit x window in artifacts/003_unit_metrics.parquet.
No unit selection is applied, so the time course is free of the selection bias of step 001.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_tables, binned_counts, baseline_rates, T_EDGES, BIN_S, PROJECT, ARTIFACTS

SLIDING = [(s, s + 0.1) for s in np.round(np.arange(-0.6, 0.55, 0.05), 2)]        # 100 ms, step 50 ms
LONGER = [(s, round(s + L, 2)) for L in (0.2, 0.3, 0.5) for s in (-0.2, -0.1, 0.0, 0.1)]
WINDOWS = SLIDING + LONGER
N_PERM = 1000
ALPHA = 0.05
MIN_UNITS = 5
OUT = ARTIFACTS / "003_unit_metrics.parquet"

def session_metrics(counts, trials_pid, rng):
    """Standardized left-right differences and permutation p-values for all units and windows.
    counts: (n_units, n_trials, n_bins). Returns dict of arrays (n_windows, n_units) for
    d_raw, p_raw, d_bl, p_bl and (n_units,) for d_base, p_base. Units silent in a window get NaN."""
    left = (trials_pid.side == "left").to_numpy()
    n_tr = len(left)
    csum = np.concatenate([np.zeros_like(counts[:, :, :1]), np.cumsum(counts, axis=2)], axis=2)   # (u, t, bins+1)
    i0 = np.searchsorted(T_EDGES, [w[0] for w in WINDOWS]); i1 = np.searchsorted(T_EDGES, [w[1] for w in WINDOWS])
    length = np.array([w[1] - w[0] for w in WINDOWS])
    rate_w = (csum[:, :, i1] - csum[:, :, i0]).transpose(2, 0, 1) / length[:, None, None]      # (w, u, t) Hz
    base = baseline_rates(counts, trials_pid)                                                    # (u, t)
    # permutation matrix: row 0 = true labels, rows 1.. = shuffled labels (same for every unit)
    labels = np.vstack([left, np.array([rng.permutation(left) for _ in range(N_PERM)])]).astype(float)  # (p+1, t)
    def d_and_p(x):
        """x: (..., t) per-trial values -> standardized diff (...,) and permutation p (...,)."""
        m_l = x @ labels.T / left.sum(); m_r = x @ (1 - labels).T / (~left).sum()           # (..., p+1)
        d = (m_l - m_r) / x.std(-1, keepdims=True)
        p = (np.abs(d[..., 1:]) >= np.abs(d[..., :1])).mean(-1)
        return d[..., 0], p
    d_raw, p_raw = d_and_p(rate_w)
    d_bl, p_bl = d_and_p(rate_w - base[None])
    d_base, p_base = d_and_p(base)
    return dict(d_raw=d_raw, p_raw=p_raw, d_bl=d_bl, p_bl=p_bl, d_base=d_base, p_base=p_base)

units, trials, split = load_tables()
if not OUT.exists():
    rng = np.random.default_rng(0)
    pids = sorted(units.loc[units.eid.isin(split.eid[split.set == "exploration"]), "pid"].unique())
    rows, t0 = [], time.time()
    for k, pid in enumerate(pids):
        counts, u, t = binned_counts(pid, units, trials)
        m = session_metrics(counts, t, rng)
        for iw, w in enumerate(WINDOWS):
            rows.append(pd.DataFrame({"pid": pid, "eid": u.eid, "subject": u.subject, "cluster_id": u.cluster_id,
                                      "region": u.beryl_acronym, "w_start": w[0], "w_end": w[1],
                                      "n_left": (t.side == "left").sum(), "n_right": (t.side == "right").sum(),
                                      "d_raw": m["d_raw"][iw], "p_raw": m["p_raw"][iw], "d_bl": m["d_bl"][iw],
                                      "p_bl": m["p_bl"][iw], "d_base": m["d_base"], "p_base": m["p_base"]}))
        if k % 10 == 0: print(f"{k + 1}/{len(pids)} insertions, {time.time() - t0:.0f} s elapsed", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(OUT)
um = pd.read_parquet(OUT)
um["w_len"] = (um.w_end - um.w_start).round(2)
print(f"{um.pid.nunique()} insertions, {um.groupby(['pid','cluster_id']).ngroups} units; "
      f"units with NaN d (silent in some window): {um.d_bl.isna().groupby([um.pid, um.cluster_id]).any().sum()}")

# ---- session x region x window fractions of significant units (replicate grain)
sr = (um.dropna(subset=["d_bl", "d_raw"]).groupby(["eid", "region", "w_start", "w_len"])
        .agg(n_units=("cluster_id", "size"), frac_bl=("p_bl", lambda p: (p < ALPHA).mean()),
             frac_raw=("p_raw", lambda p: (p < ALPHA).mean()), frac_base=("p_base", lambda p: (p < ALPHA).mean()),
             abs_d_bl=("d_bl", lambda d: np.abs(d).mean()), abs_d_raw=("d_raw", lambda d: np.abs(d).mean()))
        .reset_index())
sr = sr[sr.n_units >= MIN_UNITS]
sr.to_csv(PROJECT / "exploratory-analyses/003_session_region_window.csv", index=False)
sliding = sr[sr.w_len == 0.1]
top_regions = sliding.groupby("region").eid.nunique().sort_values(ascending=False)
top_regions = top_regions[top_regions >= 5].index[:12]
print("regions with >= 5 exploration sessions:", list(top_regions))

fig, axes = plt.subplots(2, 3, figsize=(21, 11))
# (a) time course of fraction significant, baseline-subtracted vs raw, all eligible session-regions
ax = axes[0, 0]
for col, lab in (("frac_bl", "baseline-subtracted"), ("frac_raw", "raw"), ("frac_base", "baseline only")):
    g = sliding.groupby("w_start")[col]
    ax.errorbar(g.mean().index + 0.05, g.mean(), g.sem(), label=lab, capsize=2)
ax.axhline(ALPHA, c="k", ls="--", lw=0.8, label="chance (alpha)"); ax.axvline(0, c="k", lw=0.7)
ax.set_xlabel("window centre, 100 ms windows (s from movement onset)"); ax.set_ylabel("fraction of units p < 0.05")
ax.set_title(f"All {sliding.groupby(['eid','region']).ngroups} eligible session-regions (mean +/- SEM)"); ax.legend()
# (b) same per region, baseline-subtracted
ax = axes[0, 1]
for reg in top_regions:
    g = sliding[sliding.region == reg].groupby("w_start").frac_bl.mean()
    ax.plot(g.index + 0.05, g, label=f"{reg} (n={sliding[sliding.region == reg].eid.nunique()})")
ax.axhline(ALPHA, c="k", ls="--", lw=0.8); ax.axvline(0, c="k", lw=0.7)
ax.set_xlabel("window centre (s)"); ax.set_ylabel("fraction significant (baseline-subtracted)"); ax.legend(fontsize=7, ncol=2)
ax.set_title("Per region, session-averaged")
# (c) heatmap region x window centre
ax = axes[0, 2]
hm = sliding[sliding.region.isin(top_regions)].groupby(["region", "w_start"]).frac_bl.mean().unstack()
hm = hm.loc[hm.max(1).sort_values(ascending=False).index]
im = ax.imshow(hm.values, aspect="auto", cmap="viridis", extent=[hm.columns[0], hm.columns[-1] + 0.1, len(hm), 0])
ax.set_yticks(np.arange(len(hm)) + 0.5); ax.set_yticklabels(hm.index); ax.axvline(0, c="w", lw=0.7)
ax.set_xlabel("window (s from movement onset)"); plt.colorbar(im, ax=ax, label="fraction significant (bl)")
ax.set_title("Time course by region")
# (d) window length x start
ax = axes[1, 0]
for L in (0.1, 0.2, 0.3, 0.5):
    g = sr[(sr.w_len == L) & sr.w_start.isin([-0.2, -0.1, 0.0, 0.1])].groupby("w_start").frac_bl
    ax.errorbar(g.mean().index, g.mean(), g.sem(), marker="o", capsize=2, label=f"length {L} s")
ax.axhline(ALPHA, c="k", ls="--", lw=0.8); ax.set_xlabel("window start (s)"); ax.set_ylabel("fraction significant (bl)")
ax.set_title("Window length vs start"); ax.legend()
# (e) raw vs baseline-subtracted per session-region at W = [0, 0.2]
ax = axes[1, 1]
w02 = sr[(sr.w_start == 0.0) & (sr.w_len == 0.2)]
ax.scatter(w02.frac_raw, w02.frac_bl, s=12, alpha=0.6); ax.plot([0, 1], [0, 1], "k--", lw=0.7)
ax.axhline(ALPHA, c="grey", lw=0.5); ax.axvline(ALPHA, c="grey", lw=0.5)
ax.set_xlabel("fraction significant, raw"); ax.set_ylabel("fraction significant, baseline-subtracted")
ax.set_title("Session-regions at W = [0, 0.2] s"); ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 1)
# (f) drift candidates: units significant raw & baseline but not bl, by region, at W=[0,0.2]
ax = axes[1, 2]
u02 = um[(um.w_start == 0.0) & (um.w_len == 0.2) & um.region.isin(top_regions)].dropna(subset=["d_bl"])
cls = pd.DataFrame({"region": u02.region,
                    "choice cell (bl sig)": u02.p_bl < ALPHA,
                    "drift candidate (raw & base sig, bl not)": (u02.p_raw < ALPHA) & (u02.p_base < ALPHA) & (u02.p_bl >= ALPHA)})
cls.groupby("region").mean().loc[hm.index].plot.barh(ax=ax); ax.axvline(ALPHA, c="k", ls="--", lw=0.8)
ax.invert_yaxis(); ax.set_xlabel("fraction of units"); ax.set_title("Unit classes at W = [0, 0.2] s"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(PROJECT / "exploratory-analyses/003_window_grid.png", dpi=110)

best = sliding.groupby("w_start").frac_bl.mean()
print("sliding-window fraction significant (bl), by window start:\n", best.round(3).to_string())
print("by length at start 0.0:\n", sr[sr.w_start == 0.0].groupby("w_len").frac_bl.mean().round(3).to_string())
print(cls.groupby("region").mean().round(3).to_string())
