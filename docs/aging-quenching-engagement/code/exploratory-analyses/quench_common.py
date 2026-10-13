"""Shared loading and selection for the quench × engagement project.

Selections follow Zang et al. (2026) where the package allows it (see question.md): paper ROIs from Beryl areas,
good units (label == 1, already applied in the package), task-window firing rate > 1 Hz and presence > 0.95, trials
with all events present and RT (response − stimOn) in 0.08–2 s. All trials are kept (user, 2026-10-13); the paper's
first-400 cut is a sensitivity check.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025')
PROJECT = Path(__file__).resolve().parents[1]
GLMHMM = PROJECT.parents[1] / 'aging-glmhmm-engagement' / 'code' / 'artifacts'

# Beryl area -> paper ROI (authors' utils/neuron_utils.py:combine_regions, config.py:BERYL_NAMES)
BERYL_TO_ROI = {'VISa': 'PPC', 'VISam': 'PPC', 'VISp': 'VISp+pm', 'VISpm': 'VISp+pm', 'CA1': 'CA1', 'DG': 'DG',
                'PO': 'PO', 'LP': 'LP', 'APN': 'MBm', 'MRN': 'MBm', 'SCm': 'SCm', 'MOs': 'MOs', 'ACAv': 'ACA',
                'ACAd': 'ACA', 'PL': 'mPFC', 'ILA': 'mPFC', 'ORBm': 'ORB', 'ORBl': 'ORB', 'ORBvl': 'ORB',
                'TTd': 'OLF', 'DP': 'OLF', 'AON': 'OLF', 'ACB': 'ACB', 'CP': 'CP', 'LSr': 'LS', 'LSc': 'LS', 'LSv': 'LS'}
REQUIRED_EVENTS = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times', 'stimOn_times',
                   'firstMovement_times']


def load_trials():
    """Trials passing the paper's filters, joined to GLM-HMM P(engaged).

    Returns one row per trial with `signed_contrast` (right − left, in %), `rt`, `p_engaged` (NaN for no-go trials,
    which the GLM-HMM did not fit; they are removed by the choice filter anyway) and `trial_rank` (position of the
    trial within the session, 0-based, before filtering).
    """
    trials = pd.read_parquet(DATA / 'metadata' / 'trials.parquet')
    trials['trial_rank'] = trials.groupby('session_id').cumcount()
    trials['signed_contrast'] = 100 * (trials.contrastRight.fillna(0) - trials.contrastLeft.fillna(0))
    trials['rt'] = trials.response_times - trials.stimOn_times
    keep = trials[REQUIRED_EVENTS].notna().all(axis=1) & (trials.choice != 0) & trials.rt.between(0.08, 2)
    post = pd.concat([pd.read_parquet(GLMHMM / f) for f in ['005_posteriors.parquet',
                                                            '008_posteriors_confirmation.parquet']])
    return trials[keep].merge(post[['session_id', 'trial_id', 'p_engaged']], on=['session_id', 'trial_id'],
                              how='left')


def load_units():
    """Paper-ROI units passing task-window FR > 1 Hz and presence ratio > 0.95, with session, mouse, age, split."""
    units = pd.read_parquet(DATA / 'metadata' / 'units.parquet')
    units['roi'] = units.beryl_acronym.map(BERYL_TO_ROI)
    units = units[units.roi.notna() & (units.task_firing_rate > 1) & (units.task_presence_ratio > 0.95)]
    sessions = pd.read_parquet(DATA / 'metadata' / 'sessions.parquet')
    split = pd.read_csv(PROJECT / 'artifacts' / 'split.csv')
    units = units.merge(sessions[['session_id', 'subject_id', 'lab', 'age_at_session_days']], on='session_id')
    return units.merge(split[['subject_id', 'cohort', 'split']], on='subject_id')


# Sliding windows (paper: 100 ms width, 20 ms step, causal: a timepoint is the window's END relative to stimOn)
WIDTH = 0.1
WINDOW_ENDS = np.round(np.arange(-0.3, 0.6001, 0.02), 2)
PRE_END, POST_END = 0.0, 0.26   # paper: pre = [-0.1, 0], post = [0.16, 0.26] s


def load_spikes(recording_id):
    """Spike times (s, session clock) and unit_index per spike for one recording."""
    from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard
    shard = load_spike_shard(DATA / 'spikes' / recording_id)
    return shard['spike_times_seconds'], shard['spike_clusters']


def window_counts(spike_times, spike_units, unit_indices, align_times, window_ends=WINDOW_ENDS, width=WIDTH):
    """Spike counts in sliding windows around each event.

    Returns int array (n_trials, n_units, n_windows); window w spans [t + end_w - width, t + end_w).
    Vectorised by giving each unit its own time offset so one sorted array serves all units.
    """
    offset = 1e5  # longer than any session (s), so unit blocks don't overlap
    keep = np.isin(spike_units, unit_indices)
    keys = np.sort(spike_units[keep] * offset + spike_times[keep])
    ends = align_times[:, None, None] + unit_indices[None, :, None] * offset + window_ends[None, None, :]
    return np.searchsorted(keys, ends) - np.searchsorted(keys, ends - width)


def mean_subtracted_ff(counts, conditions):
    """Mean-subtracted Fano factor (Churchland et al. 2010 style, as in Zang et al. 2026).

    counts: (n_trials, n_units, n_windows); conditions: (n_trials,) signed contrast labels.
    Residual = count minus its condition mean; FF = variance of pooled residuals / pooled mean count
    (the trial-weighted average of condition means equals the pooled mean). Returns (n_units, n_windows).
    Deviation from the paper: residual variance divides by (n_trials − n_conditions), not n_trials − 1, because
    each condition mean uses one degree of freedom. Negligible at ~400 trials, but it matters for the small
    trial-matched subsamples (e.g. 50 trials over 9 conditions) used to compare engaged with disengaged trials.
    """
    labels, index = np.unique(conditions, return_inverse=True)
    condition_mean = np.stack([counts[index == k].mean(0) for k in range(len(labels))])
    residuals = counts - condition_mean[index]
    return (residuals ** 2).sum(0) / (len(conditions) - len(labels)) / counts.mean(0)


def state_metrics(counts, conditions, window_ends=WINDOW_ENDS, post_end=POST_END):
    """Pre/post FR (Hz), pre/post FF and quench (post FF − pre FF) per unit for one trial subset.

    counts: (n_trials, n_units, n_windows). post_end: end of the post window (paper 0.26 s; movement-safe 0.12 s).
    Returns dict of (n_units,) arrays.
    """
    ff = mean_subtracted_ff(counts, conditions)
    fr = counts.mean(0) / WIDTH
    pre, post = np.searchsorted(window_ends, PRE_END), np.searchsorted(window_ends, post_end)
    return {'fr_pre': fr[:, pre], 'fr_post': fr[:, post], 'ff_pre': ff[:, pre], 'ff_post': ff[:, post],
            'quench': ff[:, post] - ff[:, pre]}


PARTB_VARIANTS = {'paper': dict(post_end=0.26, move_after=-np.inf, n=150),
                  'safe': dict(post_end=0.12, move_after=0.12, n=100)}
METRICS = ['fr_pre', 'fr_post', 'ff_pre', 'ff_post', 'quench']
LOG_METRICS = ['fr_pre', 'fr_post', 'ff_pre', 'ff_post']


def partB_metrics(units, trials, theta=0.8, n_repeats=20, seed=0):
    """Per-neuron metrics on equal-size random sets of engaged ('eng') and any-state ('all') trials.

    Same computation as 004_age_effects_exploration.py. For each window variant in PARTB_VARIANTS, trials with
    first movement <= move_after are dropped, sessions with fewer than n engaged trials are skipped, and metrics are
    averaged over n_repeats draws of n trials. Returns a long DataFrame (neuron × variant × trial_set).
    """
    rng = np.random.default_rng(seed)
    trials = trials[trials.session_id.isin(units.session_id)].sort_values(['session_id', 'trial_id'])
    mv = trials.firstMovement_times - trials.stimOn_times
    rows = []
    for session_id, st in trials.groupby('session_id'):
        su = pd.concat([u for _, u in units[units.session_id == session_id].groupby('recording_id')])
        for variant, v in PARTB_VARIANTS.items():
            windows = np.array([PRE_END, v['post_end']])
            sv = st[mv[st.index] > v['move_after']]
            pools = {'eng': np.flatnonzero(sv.p_engaged.values > theta), 'all': np.arange(len(sv))}
            if len(pools['eng']) < v['n']:
                continue
            counts = np.concatenate([window_counts(*load_spikes(rec), u.unit_index.values, sv.stimOn_times.values,
                                                   windows) for rec, u in su.groupby('recording_id')], axis=1)
            for trial_set, pool in pools.items():
                draws = [state_metrics(counts[d], sv.signed_contrast.values[d], windows, v['post_end'])
                         for d in (rng.choice(pool, v['n'], replace=False) for _ in range(n_repeats))]
                m = {k: np.mean([d[k] for d in draws], 0) for k in METRICS}
                rows.append(su[['session_id', 'recording_id', 'unit_index', 'roi', 'subject_id', 'lab', 'cohort',
                                'age_at_session_days']].assign(variant=variant, trial_set=trial_set, **m))
    return pd.concat(rows, ignore_index=True)


def age_slopes(d, metric, ages):
    """β_age with ROI fixed effects for many age assignments at once (Frisch–Waugh: residualise on ROI means).

    d: neuron rows (columns subject_id, roi, metric); ages: (n_sets, n_mice) mouse ages in 100 d, columns ordered as
    np.unique(d.subject_id). Log scale for LOG_METRICS. Returns (n_sets,) slopes.
    """
    y = np.log(d[metric].values) if metric in LOG_METRICS else d[metric].values
    roi = pd.factorize(d.roi)[0]
    mouse = np.unique(d.subject_id, return_inverse=True)[1]
    n_roi = np.bincount(roi)
    y_res = y - (np.bincount(roi, y) / n_roi)[roi]
    a = ages[:, mouse]                                            # (n_sets, n_neurons)
    a_roi_mean = np.stack([np.bincount(roi, row) / n_roi for row in a])
    a_res = a - a_roi_mean[:, roi]
    return (a_res @ y_res) / (a_res ** 2).sum(1)
