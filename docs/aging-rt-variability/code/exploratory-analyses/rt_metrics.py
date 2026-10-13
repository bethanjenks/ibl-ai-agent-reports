"""Shared per-session RT metrics, so every script computes them identically."""
import numpy as np
import pandas as pd

MIN_TRIALS_PER_CONTRAST = 10


def spread_metrics(rt):
    """Location and spread of one set of RTs (1-D array, seconds). Returns a dict of scalars."""
    q25, q50, q75 = np.quantile(rt, [0.25, 0.5, 0.75])
    return {'median_rt': q50,
            'cv': rt.std(ddof=1) / rt.mean(),                         # SD / mean; sensitive to the tails
            'robust_cv': (q75 - q25) / q50,                           # IQR / median; ignores the outer 50%
            'sd_log_rt': np.log(rt).std(ddof=1)}                      # spread on a log scale (≈ CV for lognormal RTs)


def session_metrics(trials, window=(0.08, 2.0), drop_early=False, first_n=400, fast=0.08, slow=2.0):
    """Per-session RT metrics.

    trials: per-trial table from 001 (needs session_id, trial_id, complete, rt, contrast, early_move).
    Kept trials = first `first_n`, complete, RT inside `window` (and no wheel movement before stimOn if drop_early).
    Pooled metrics use all kept trials. `pc_*` metrics are computed within each unsigned contrast (≥ 10 trials)
    and averaged, so a session's contrast mix cannot change them. Tail rates are fractions of complete first_n
    trials with RT < `fast` or > `slow` seconds, independent of `window`.
    Returns one row per session.
    """
    complete = trials[(trials.trial_id < first_n) & trials.complete]
    kept = complete[complete.rt.between(*window)]
    if drop_early:
        kept = kept[~kept.early_move]
    pooled = kept.groupby('session_id').rt.apply(lambda r: pd.Series(spread_metrics(r.values))).unstack()
    per_contrast = (kept.groupby(['session_id', 'contrast']).rt
                    .apply(lambda r: pd.Series(spread_metrics(r.values)) if len(r) >= MIN_TRIALS_PER_CONTRAST else None)
                    .unstack().groupby('session_id').mean().add_prefix('pc_'))
    tails = complete.groupby('session_id').rt.agg(frac_fast=lambda r: np.mean(r < fast), frac_slow=lambda r: np.mean(r > slow))
    out = pooled.join(per_contrast).join(tails)
    out['n_kept'] = kept.groupby('session_id').size()
    info = trials.groupby('session_id')[['subject_id', 'cohort', 'lab', 'age_at_session_days']].first()
    return info.join(out).reset_index()


def mouse_metrics(sessions):
    """Average session metrics within mouse (the independent unit). Returns one row per mouse."""
    num = sessions.select_dtypes('number').columns
    m = sessions.groupby('subject_id')[list(num)].mean()
    m['n_sessions'] = sessions.groupby('subject_id').size()
    return sessions.groupby('subject_id')[['cohort', 'lab']].first().join(m).reset_index()
