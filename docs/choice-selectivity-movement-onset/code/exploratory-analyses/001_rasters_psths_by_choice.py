"""Single-unit sanity plots: rasters and PSTHs around movement onset, split by choice.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

Three exploration insertions (thalamus/visual cortex, motor cortex, midbrain/hippocampus).
For each, the two units with the largest standardized left-right difference in [-0.1, 0.1] s
plus one random unit are shown, so both clear and typical cases are visible.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from common import load_tables, binned_counts, T_CENTERS, BIN_S, PROJECT

PIDS = ["0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a",   # PO / LP / VPM / VISa (repeated site)
        "0e8d3cb3-2e4c-47a0-8bb4-65d2cc3161c6",   # MOp
        "02cc03e4-8015-4050-bb42-6c832091febb"]   # APN / MRN / VISa / CA1
WIN = (-0.1, 0.1)          # provisional "around movement onset" window for ranking units
SMOOTH_BINS = 2            # 20 ms Gaussian smoothing of PSTHs for display only
rng = np.random.default_rng(0)

units, trials, split = load_tables()
in_win = (T_CENTERS >= WIN[0]) & (T_CENTERS < WIN[1])

fig, axes = plt.subplots(3, 9, figsize=(26, 11), gridspec_kw={"height_ratios": [1, 1, 1]})
for r, pid in enumerate(PIDS):
    counts, u, t = binned_counts(pid, units, trials)
    left = (t.side == "left").to_numpy()
    rate_win = counts[:, :, in_win].sum(2) / (WIN[1] - WIN[0])           # (n_units, n_trials) Hz
    # standardized difference: (mean_L - mean_R) / pooled sd across trials
    d = (rate_win[:, left].mean(1) - rate_win[:, ~left].mean(1)) / (rate_win.std(1) + 1e-9)
    picks = list(np.argsort(-np.abs(d))[:2]) + [rng.integers(len(u))]
    for c, iu in enumerate(picks):
        ax_r, ax_p = axes[r, 3 * c], axes[r, 3 * c + 1]
        order = np.concatenate([np.where(left)[0], np.where(~left)[0]])   # left trials on top
        for k, it in enumerate(order):
            spikes = T_CENTERS[np.repeat(np.arange(len(T_CENTERS)), counts[iu, it])]
            ax_r.scatter(spikes, np.full(len(spikes), k), s=1.5, c="C0" if left[it] else "C1", marker="|")
        ax_r.axvline(0, c="k", lw=0.7); ax_r.set_xlim(-0.5, 0.5); ax_r.set_ylabel("trial (L then R)")
        ax_r.set_title(f"{u.beryl_acronym[iu]} cluster {u.cluster_id[iu]}  d={d[iu]:+.2f}", fontsize=9)
        for side, m in (("left", left), ("right", ~left)):
            psth = gaussian_filter1d(counts[iu, m].mean(0) / BIN_S, SMOOTH_BINS)
            sem = gaussian_filter1d(counts[iu, m].std(0) / BIN_S / np.sqrt(m.sum()), SMOOTH_BINS)
            ax_p.plot(T_CENTERS, psth, label=f"{side} (n={m.sum()})"); ax_p.fill_between(T_CENTERS, psth - sem, psth + sem, alpha=0.3)
        ax_p.axvline(0, c="k", lw=0.7); ax_p.axvspan(*WIN, color="grey", alpha=0.2); ax_p.set_xlim(-0.5, 0.5)
        ax_p.set_ylabel("rate (Hz)"); ax_p.legend(fontsize=7); ax_p.set_xlabel("time from movement onset (s)")
    # insertion overview: every unit's mean rate in the window, left vs right
    ax = axes[r, 2]
    for reg in u.beryl_acronym.value_counts().index[:5]:
        m = (u.beryl_acronym == reg).to_numpy()
        ax.scatter(rate_win[m][:, ~left].mean(1), rate_win[m][:, left].mean(1), s=10, label=reg)
    lim = rate_win.mean(1).max() * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.7); ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel("right-choice rate (Hz)"); ax.set_ylabel("left-choice rate (Hz)"); ax.legend(fontsize=7)
    ax.set_title(f"{len(u)} units, {len(t)} trials", fontsize=9)
    for c in (5, 8): axes[r, c].axis("off")
    print(pid, u.subject[0], f"{len(u)} units, {len(t)} trials ({left.sum()} L / {(~left).sum()} R); "
          f"|d| > 0.5 in {np.mean(np.abs(d) > 0.5):.0%} of units; top regions {dict(u.beryl_acronym.value_counts().head(4))}")
fig.tight_layout()
fig.savefig(PROJECT / "exploratory-analyses/001_rasters_psths_by_choice.png", dpi=110)
