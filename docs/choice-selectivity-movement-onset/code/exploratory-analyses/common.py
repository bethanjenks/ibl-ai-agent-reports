"""Shared helpers: movement-onset-aligned spike counts for unbiased-block trials, cached per insertion."""
import numpy as np
import pandas as pd
from scipy.stats import binom, wilcoxon, false_discovery_control
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard

DATASET_DIR = resolve_dataset_dir("bwm_ephys")   # bwm_ephys 1.2.1
PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts"
BIN_S = 0.01                                     # 10 ms bins
T_EDGES = np.round(np.arange(-2.5, 1.0 + BIN_S / 2, BIN_S), 3)   # -2.5 to +1 s around movement onset;
# -2.5 s reaches the pre-stimulus baseline on every trial because the BWM mask caps reaction time at 2 s
T_CENTERS = T_EDGES[:-1] + BIN_S / 2

def load_tables():
    """Return (units, trials, split): good units with a Beryl label, analysable unbiased-block
    trials from step 000, and the session-level exploration/confirmation split."""
    units = pd.read_parquet(DATASET_DIR / "metadata/units.parquet").dropna(subset=["beryl_acronym"])
    trials = pd.read_parquet(ARTIFACTS / "unbiased_trials.parquet")
    split = pd.read_csv(ARTIFACTS / "data_split.csv")
    return units, trials, split

def binned_counts(pid, units, trials, cache_dir=ARTIFACTS / "binned"):
    """Spike counts of every good unit of insertion `pid` in 10 ms bins from -2.5 to +1 s around
    firstMovement_times of each analysable unbiased-block trial.

    Returns (counts, units_pid, trials_pid): counts is int16 (n_units, n_trials, n_bins); rows of
    units_pid / trials_pid give the unit and trial for each axis. Cached in cache_dir/<pid>.npz unless cache_dir is None.
    Assumes the shard contains only good units and that the trial table is already filtered."""
    units_pid = units[units.pid == pid].sort_values("cluster_id").reset_index(drop=True)
    trials_pid = trials[trials.eid == units_pid.eid.iloc[0]].sort_values("trial_id").reset_index(drop=True)
    cache = None if cache_dir is None else cache_dir / f"{pid}.npz"
    if cache is not None and cache.exists():
        z = np.load(cache)
        assert np.array_equal(z["cluster_ids"], units_pid.cluster_id) and np.array_equal(z["trial_ids"], trials_pid.trial_id)
        return z["counts"], units_pid, trials_pid
    shard = load_spike_shard(DATASET_DIR / "spikes" / pid)
    spike_cluster = shard["cluster_ids"][shard["spike_clusters"]]     # dense index -> real cluster id
    spike_t = shard["spike_times_seconds"]
    unit_row = pd.Series(np.arange(len(units_pid)), index=units_pid.cluster_id)
    keep = np.isin(spike_cluster, units_pid.cluster_id)
    spike_t, spike_row = spike_t[keep], unit_row[spike_cluster[keep]].to_numpy()
    onsets = trials_pid.firstMovement_times.to_numpy()
    # A spike can fall in the window of the last onset before it or of the next onset
    # after it (windows of consecutive trials overlap when trials are < 3.5 s apart), so try both.
    counts = np.zeros((len(units_pid), len(trials_pid), len(T_CENTERS)), dtype=np.int16)
    i_prev = np.searchsorted(onsets, spike_t) - 1
    for i_trial in (i_prev, i_prev + 1):
        valid = (i_trial >= 0) & (i_trial < len(onsets))
        rel = spike_t - onsets[np.clip(i_trial, 0, len(onsets) - 1)]
        in_win = valid & (rel >= T_EDGES[0]) & (rel < T_EDGES[-1])
        i_bin = ((rel[in_win] - T_EDGES[0]) / BIN_S).astype(int)
        np.add.at(counts, (spike_row[in_win], i_trial[in_win], i_bin), 1)
    if cache is not None:
        cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache, counts=counts, cluster_ids=units_pid.cluster_id.to_numpy(), trial_ids=trials_pid.trial_id.to_numpy())
    return counts, units_pid, trials_pid

BASELINE_S = 0.3     # pre-stimulus baseline length; the mouse must hold the wheel still before stimulus onset

def window_rates(counts, trials_pid, window):
    """Mean firing rate (Hz) of each unit on each trial in `window` (start, end) s relative to
    movement onset. counts: (n_units, n_trials, n_bins). Returns (n_units, n_trials)."""
    m = (T_CENTERS >= window[0]) & (T_CENTERS < window[1])
    return counts[:, :, m].sum(2) / (window[1] - window[0])

def baseline_rates(counts, trials_pid):
    """Mean firing rate (Hz) of each unit in the BASELINE_S seconds before stimulus onset of each
    trial, computed from movement-aligned bins (stimulus onset is at -reaction time).
    Returns (n_units, n_trials)."""
    t_stim = (trials_pid.stimOn_times - trials_pid.firstMovement_times).to_numpy()   # negative
    assert (t_stim - BASELINE_S >= T_EDGES[0]).all()
    in_base = (T_CENTERS[None, :] >= (t_stim - BASELINE_S)[:, None]) & (T_CENTERS[None, :] < t_stim[:, None])
    return (counts * in_base[None]).sum(2) / BASELINE_S

WIDE = (-0.2, 0.2)                                                   # provisional analysis window
SUB_EDGES = np.round(np.arange(WIDE[0], WIDE[1] + 0.05, 0.1), 2)     # 4 x 100 ms sub-bins

def subbin_rates(counts, edges):
    """Per-unit, per-trial mean rate (Hz) in each [edges[k], edges[k+1]) -> (K, n_units, n_trials)."""
    csum = np.concatenate([np.zeros_like(counts[:, :, :1]), np.cumsum(counts, axis=2)], axis=2)
    i = np.searchsorted(T_EDGES, edges)
    return (csum[:, :, i[1:]] - csum[:, :, i[:-1]]).transpose(2, 0, 1) / np.diff(edges)[:, None, None]

def standardized_diffs(x, labels):
    """x: (..., n_trials) per-trial values; labels: (n_perm + 1, n_trials) boolean-as-float with
    the true labels in row 0. Returns (mean_1 - mean_0) / SD for every row -> (..., n_perm + 1)."""
    n1 = labels[0].sum(); n0 = len(labels[0]) - n1
    return (x @ labels.T / n1 - x @ (1 - labels).T / n0) / x.std(-1, keepdims=True)

def omnibus_stats(counts, trials_pid, labels, edges=SUB_EDGES):
    """Omnibus statistic for every unit and every label row: sum over sub-bins of squared
    baseline-subtracted standardized differences. counts: (n_units, n_trials, n_bins);
    labels: (n_perm + 1, n_trials) with true labels in row 0. Returns (stat (n_units, n_perm + 1),
    d_sub (K, n_units) true-label differences). Silent sub-bins contribute 0."""
    sub = subbin_rates(counts, edges) - baseline_rates(counts, trials_pid)[None]          # (K, u, t)
    d = standardized_diffs(sub, labels)                                                   # (K, u, p+1)
    return np.nansum(d ** 2, axis=0), d[:, :, 0]

def omnibus_test(counts, trials_pid, labels, edges=SUB_EDGES):
    """Per-unit omnibus statistic, permutation p-value and true-label sub-bin differences.
    Returns (stat (n_units,), p (n_units,), d_sub (K, n_units))."""
    stat, d_sub = omnibus_stats(counts, trials_pid, labels, edges)
    return stat[:, 0], (stat[:, 1:] >= stat[:, :1]).mean(1), d_sub

def permuted_labels(true, rng, n_perm, strata=None):
    """(n_perm + 1, n_trials) float array: row 0 = true labels; other rows shuffle labels, within
    each stratum if `strata` (n_trials,) is given."""
    out = np.empty((n_perm + 1, len(true)), dtype=float); out[0] = true
    groups = [np.arange(len(true))] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    for i in range(1, n_perm + 1):
        for idx in groups:
            out[i, idx] = rng.permutation(true[idx])
    return out

def session_region_fractions(units, trials, eids, edges, n_perm=1000, alpha=0.05, min_units=5):
    """For every session in `eids` x region with >= min_units units: fraction of units
    significant under the true labels and under each of n_perm shared label permutations.
    Returns DataFrame(eid, region, n_units) and array f (n_rows, n_perm + 1)."""
    keys, f = [], []
    for eid, units_eid in units[units.eid.isin(eids)].groupby("eid"):
        stats, regions = [], []
        for pid in units_eid.pid.unique():
            counts, u, t = binned_counts(pid, units, trials)
            labels = permuted_labels((t.side == "left").to_numpy(), np.random.default_rng(int(eid.replace("-", ""), 16) % 2**32), n_perm)
            s, _ = omnibus_stats(counts, t, labels, edges)                       # (u, p+1), same labels for all probes
            stats.append(s); regions.append(u.beryl_acronym.to_numpy())
        stats, regions = np.vstack(stats), np.concatenate(regions)
        thr = np.nanquantile(stats[:, 1:], 1 - alpha, axis=1, keepdims=True)    # per-unit null 95th percentile
        sig = np.nan_to_num(stats) > thr                                        # (u, p+1) "significant" under each labeling
        for reg in np.unique(regions):
            m = regions == reg
            if m.sum() >= min_units:
                keys.append((eid, reg, m.sum())); f.append(sig[m].mean(0))
    return pd.DataFrame(keys, columns=["eid", "region", "n_units"]), np.array(f)

def region_tests(keys, f, rng, n_perm=1000, alpha=0.05, min_sessions=5, n_draws=10000):
    """Three one-sided p-values per region with >= min_sessions sessions, plus effect sizes and
    BH q-values for each test separately (q_binom, q_wilcoxon, q_sessperm)."""
    out = []
    for reg, idx in keys.groupby("region").indices.items():
        if len(idx) < min_sessions: continue
        fo, n = f[idx, 0], keys.n_units.to_numpy()[idx]
        k = int(round((fo * n).sum()))
        p_binom = binom.sf(k - 1, n.sum(), alpha)
        p_wilc = wilcoxon(fo - alpha, alternative="greater", zero_method="zsplit").pvalue if np.any(fo != alpha) else 1.0
        null = np.stack([f[i, rng.integers(1, n_perm + 1, n_draws)] for i in idx]).mean(0)   # (n_draws,)
        p_sp = ((null >= fo.mean()).sum() + 1) / (n_draws + 1)
        out.append(dict(region=reg, n_sessions=len(idx), n_units=n.sum(), mean_frac=fo.mean(), sem_frac=fo.std(ddof=1) / np.sqrt(len(idx)),
                        null_mean=null.mean(), p_binom=p_binom, p_wilcoxon=p_wilc, p_sessperm=p_sp))
    out = pd.DataFrame(out).sort_values("mean_frac", ascending=False)
    for c in ("p_binom", "p_wilcoxon", "p_sessperm"):
        out[c.replace("p_", "q_")] = false_discovery_control(out[c], method="bh")
    return out

