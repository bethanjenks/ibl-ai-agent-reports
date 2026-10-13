"""Metric-definition figure: how a unit's choice selectivity is reduced to scalars.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

Per unit and trial we take the firing rate in a movement-onset window W (provisional 0 to
+0.2 s; gridded in step 003) and in a 0.3 s pre-stimulus baseline B. Three scalars follow:
  delta_raw  = mean_L(W) - mean_R(W)                       raw window difference
  delta_bl   = mean_L(W - B) - mean_R(W - B)               baseline-subtracted difference
  delta_base = mean_L(B) - mean_R(B)                       baseline difference (drift indicator)
Each is standardized by the pooled across-trial SD and tested with a choice-label permutation.
Shown for a movement-locked APN unit and a VISa unit whose left-right offset spans the whole trial.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from common import load_tables, binned_counts, window_rates, baseline_rates, T_CENTERS, BIN_S, BASELINE_S, PROJECT

EXAMPLES = [("02cc03e4-8015-4050-bb42-6c832091febb", 344),   # APN, movement-locked
            ("0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a", 733)]   # VISa, whole-trial offset
WIN = (0.0, 0.2)
N_PERM = 5000
rng = np.random.default_rng(0)

def standardized_diff(x, left):
    """(mean_L - mean_R) / pooled SD of the per-trial values x (n_trials,)."""
    return (x[left].mean() - x[~left].mean()) / x.std()

def perm_test(x, left):
    """Two-sided permutation test of standardized_diff by shuffling choice labels.
    Returns (observed, null (N_PERM,), p)."""
    obs = standardized_diff(x, left)
    null = np.array([standardized_diff(x, rng.permutation(left)) for _ in range(N_PERM)])
    return obs, null, (np.abs(null) >= abs(obs)).mean()

units, trials, split = load_tables()
fig, axes = plt.subplots(2, 5, figsize=(26, 9.5))
for r, (pid, cid) in enumerate(EXAMPLES):
    counts, u, t = binned_counts(pid, units, trials)
    iu = int(np.where(u.cluster_id == cid)[0][0])
    left = (t.side == "left").to_numpy()
    w, b = window_rates(counts, t, WIN)[iu], baseline_rates(counts, t)[iu]
    t_stim = (t.stimOn_times - t.firstMovement_times).to_numpy()
    title = f"{u.beryl_acronym[iu]} cluster {cid}"

    # (1) PSTH by choice with the window and the spread of per-trial baseline periods
    ax = axes[r, 0]
    for side, m in (("left", left), ("right", ~left)):
        ax.plot(T_CENTERS, gaussian_filter1d(counts[iu, m].mean(0) / BIN_S, 2), label=f"{side} (n={m.sum()})")
    ax.axvspan(*WIN, color="grey", alpha=0.3, label="window W")
    ax.axvspan(np.median(t_stim) - BASELINE_S, np.median(t_stim), color="green", alpha=0.2, label="baseline B (median trial)")
    ax.axvline(0, c="k", lw=0.7); ax.set_xlim(-1.2, 0.6); ax.set_xlabel("time from movement onset (s)"); ax.set_ylabel("rate (Hz)")
    ax.legend(fontsize=8); ax.set_title(title, fontsize=10)

    # (2) per-trial baseline vs window rate, coloured by choice
    ax = axes[r, 1]
    ax.scatter(b[left], w[left], s=14, label="left"); ax.scatter(b[~left], w[~left], s=14, label="right")
    lim = max(b.max(), w.max()) * 1.05; ax.plot([0, lim], [0, lim], "k--", lw=0.7)
    ax.set_xlabel("baseline rate B (Hz)"); ax.set_ylabel("window rate W (Hz)"); ax.legend(fontsize=8)
    ax.set_title("one dot per trial", fontsize=10)

    # (3-5) the three scalars: per-trial distributions and permutation nulls
    for c, (name, x) in enumerate((("delta_raw: W", w), ("delta_bl: W - B", w - b), ("delta_base: B", b))):
        obs, null, p = perm_test(x, left)
        ax = axes[r, 2 + c]
        bins = np.linspace(x.min(), x.max(), 16)
        ax.hist(x[left], bins, alpha=0.6, label=f"left, mean {x[left].mean():.1f}")
        ax.hist(x[~left], bins, alpha=0.6, label=f"right, mean {x[~left].mean():.1f}")
        ax.set_xlabel("per-trial value (Hz)"); ax.set_ylabel("trials"); ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{name}\nstandardized d = {obs:+.2f}, permutation p = {p:.4f}", fontsize=10)
        ins = ax.inset_axes([0.62, 0.55, 0.36, 0.4])
        ins.hist(null, 40, color="grey"); ins.axvline(obs, c="r"); ins.set_xticks([-1, 0, 1]); ins.set_yticks([])
        ins.set_title("label-permutation null", fontsize=7)
        print(f"{title:35s} {name:16s} d={obs:+.2f} p={p:.4f}")
fig.tight_layout()
fig.savefig(PROJECT / "exploratory-analyses/002_metric_definition.png", dpi=110)
