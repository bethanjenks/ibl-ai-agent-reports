"""Pilot over all exploration sessions: per-neuron FR/FF/quench by engagement state, with metric diagnostics.

Per session, spike counts are computed once for all filtered trials (pre and post windows only) and metrics
(`state_metrics`) are computed on trial subsets. Subsampled sets are averaged over N_REPEATS random draws.

Part A (engaged vs disengaged; sessions with >= MIN_DIS disengaged trials, for each THETA):
  dis       all disengaged trials (n = n_dis)
  eng       random engaged trials, n_dis of them
  eng_time  engaged trials nearest in trial position to each disengaged trial (time-in-session control:
            in bwm, disengagement is often one block at session end, where drift could mimic a state effect)
  placebo:  early_eng vs late_eng = first vs last n_dis engaged trials (drift with no state change)
Part B preparation (THETA = 0.8): `engB` and `allB` = N_B random engaged / all trials (same count for every
  neuron so FF bias is equal); `all_full` and `first400` = paper-like, all filtered trials / first 400 trials.
Trial-count bias: engaged FF at n in N_GRID.
Movement: fraction of trials with firstMovement before the post-window end, per state (FF is sensitive to it).

Outputs: artifacts/002_metrics_per_neuron.parquet (long: unit × condition × theta), artifacts/002_trial_info.parquet,
exploratory-analyses/002_ff_pilot.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quench_common import (PROJECT, POST_END, PRE_END, load_trials, load_units, load_spikes, window_counts,
                           state_metrics)

THETAS, MIN_DIS, N_B, N_REPEATS = [0.6, 0.8, 0.9], 20, 150, 20
N_GRID = [20, 50, 100, 150, 250]
WINDOWS = np.array([PRE_END, POST_END])
COHORT_COLORS = {'bwm': '#2a78d6', 'lifespan': '#d6552a'}
rng = np.random.default_rng(0)


def subsample_metrics(counts, contrasts, pool, n):
    """Mean of state_metrics over N_REPEATS random draws of n trials from index array `pool`."""
    draws = [state_metrics(counts[d], contrasts[d], WINDOWS)
             for d in (rng.choice(pool, n, replace=False) for _ in range(N_REPEATS))]
    return {k: np.mean([d[k] for d in draws], 0) for k in draws[0]}


def time_matched(rank, dis_idx, eng_idx):
    """For each disengaged trial, the nearest-in-position unused engaged trial (greedy, no replacement)."""
    free, chosen = list(eng_idx), []
    for i in dis_idx:
        j = int(np.argmin(np.abs(rank[free] - rank[i])))
        chosen.append(free.pop(j))
    return np.array(chosen)


units = load_units()
units = units[units.split == 'exploration']
trials = load_trials()
trials = trials[trials.session_id.isin(units.session_id)].sort_values(['session_id', 'trial_id'])
trials['moved_in_post'] = trials.firstMovement_times < trials.stimOn_times + POST_END
trials.to_parquet(PROJECT / 'artifacts' / '002_trial_info.parquet')

rows = []
for session_id, st in trials.groupby('session_id'):
    su = pd.concat([u for _, u in units[units.session_id == session_id].groupby('recording_id')])
    counts = np.concatenate([window_counts(*load_spikes(rec), u.unit_index.values, st.stimOn_times.values, WINDOWS)
                             for rec, u in su.groupby('recording_id')], axis=1)
    contrasts, rank, p = st.signed_contrast.values, st.trial_rank.values, st.p_engaged.values
    results = []  # (theta, condition, n, metrics)
    eng = np.flatnonzero(p > 0.8)
    results.append((np.nan, 'all_full', len(st), state_metrics(counts, contrasts, WINDOWS)))
    first = np.flatnonzero(rank < 400)
    results.append((np.nan, 'first400', len(first), state_metrics(counts[first], contrasts[first], WINDOWS)))
    if len(eng) >= N_B:
        results.append((0.8, 'engB', N_B, subsample_metrics(counts, contrasts, eng, N_B)))
        results.append((0.8, 'allB', N_B, subsample_metrics(counts, contrasts, np.arange(len(st)), N_B)))
    for n in N_GRID:
        if len(eng) >= n:
            results.append((0.8, f'eng_n{n}', n, subsample_metrics(counts, contrasts, eng, n)))
    for theta in THETAS:
        eng_t, dis_t = np.flatnonzero(p > theta), np.flatnonzero(p < 1 - theta)
        n = len(dis_t)
        if n < MIN_DIS or len(eng_t) < 2 * n:
            continue
        matched = time_matched(rank, dis_t, eng_t)
        results += [(theta, 'dis', n, state_metrics(counts[dis_t], contrasts[dis_t], WINDOWS)),
                    (theta, 'eng', n, subsample_metrics(counts, contrasts, eng_t, n)),
                    (theta, 'eng_time', n, state_metrics(counts[matched], contrasts[matched], WINDOWS)),
                    (theta, 'early_eng', n, state_metrics(counts[eng_t[:n]], contrasts[eng_t[:n]], WINDOWS)),
                    (theta, 'late_eng', n, state_metrics(counts[eng_t[-n:]], contrasts[eng_t[-n:]], WINDOWS))]
    for theta, condition, n, m in results:
        rows.append(su[['session_id', 'recording_id', 'unit_index', 'roi', 'subject_id', 'lab', 'cohort',
                        'age_at_session_days']].assign(theta=theta, condition=condition, n_trials=n, **m))
    print(session_id[:8], st.subject_id.iloc[0] if 'subject_id' in st else '', len(su), 'units', flush=True)

df = pd.concat(rows, ignore_index=True)
df.to_parquet(PROJECT / 'artifacts' / '002_metrics_per_neuron.parquet')

# ---------------- diagnostics figure ----------------
METRICS = ['fr_pre', 'fr_post', 'ff_pre', 'ff_post', 'quench']
fig, ax = plt.subplots(2, 4, figsize=(22, 10))
cohort_of = units.drop_duplicates('session_id').set_index('session_id').cohort
trials['cohort'] = trials.session_id.map(cohort_of)
trials['state'] = np.select([trials.p_engaged > 0.8, trials.p_engaged < 0.2], ['engaged', 'disengaged'], 'uncertain')

# (0,0) where disengaged trials sit in the session: block at end vs interleaved
trials['rel_pos'] = trials.groupby('session_id').trial_rank.transform(lambda r: r / r.max())
for cohort, d in trials[trials.state == 'disengaged'].groupby('cohort'):
    ax[0, 0].hist(d.rel_pos, bins=20, range=(0, 1), density=True, histtype='step', lw=2, color=COHORT_COLORS[cohort],
                  label=cohort)
ax[0, 0].set(xlabel='relative position in session', ylabel='density', title='Where disengaged trials (P < 0.2) sit')
ax[0, 0].legend()

# (0,1) movement inside the post window, by state and cohort
mv = trials.groupby(['cohort', 'state']).moved_in_post.mean().unstack()
mv[['engaged', 'uncertain', 'disengaged']].plot.bar(ax=ax[0, 1], color=['#1baf7a', '#bbbbbb', '#9b59b6'])
ax[0, 1].set(ylabel=f'fraction moved before stimOn + {POST_END} s', title='Movement inside post window', xlabel='')

# (0,2) FF trial-count bias (engaged trials, session medians)
for cohort, d in df[df.condition.str.startswith('eng_n')].groupby('cohort'):
    m = d.groupby(['n_trials', 'session_id'])[['ff_pre', 'ff_post']].median().groupby('n_trials').median()
    ax[0, 2].plot(m.index, m.ff_pre, 'o-', color=COHORT_COLORS[cohort], label=f'{cohort} pre')
    ax[0, 2].plot(m.index, m.ff_post, 's--', color=COHORT_COLORS[cohort], label=f'{cohort} post')
ax[0, 2].set(xlabel='engaged trials used', ylabel='median FF (session medians)', title='FF vs trial count')
ax[0, 2].legend()

# (0,3) all-trial vs first-400 quench (paper-like), session medians
w = df[df.condition.isin(['all_full', 'first400'])].pivot_table(index=['session_id', 'unit_index', 'recording_id', 'cohort'],
                                                                 columns='condition', values='quench').reset_index()
s = w.groupby(['session_id', 'cohort'])[['all_full', 'first400']].median().reset_index()
for cohort, d in s.groupby('cohort'):
    ax[0, 3].scatter(d.first400, d.all_full, color=COHORT_COLORS[cohort], label=cohort, s=15)
lim = [s[['all_full', 'first400']].min().min(), s[['all_full', 'first400']].max().max()]
ax[0, 3].plot(lim, lim, color='grey')
ax[0, 3].set(xlabel='quench, first 400 trials', ylabel='quench, all trials', title='Session median quench')
ax[0, 3].legend()

# row 1: Part A, session-median within-neuron differences (dis − comparison) per metric, theta 0.8
a = df[df.theta == 0.8].pivot_table(index=['session_id', 'recording_id', 'unit_index', 'cohort'], columns='condition',
                                     values=METRICS)
contrasts_A = {'dis − eng': ('dis', 'eng'), 'dis − eng_time': ('dis', 'eng_time'),
               'placebo late − early': ('late_eng', 'early_eng')}
for col, metric in enumerate(['fr_pre', 'ff_pre', 'ff_post', 'quench']):
    x = 0
    for label, (c1, c2) in contrasts_A.items():
        diff = (a[(metric, c1)] - a[(metric, c2)])
        if metric.startswith('fr'):
            diff = np.log2(a[(metric, c1)] + 0.5) - np.log2(a[(metric, c2)] + 0.5)
        per_session = diff.groupby(['session_id', 'cohort']).median().dropna().reset_index(name='d')
        for i, (cohort, d) in enumerate(per_session.groupby('cohort')):
            xs = x + 0.3 * i + rng.uniform(-0.08, 0.08, len(d))
            ax[1, col].scatter(xs, d.d, s=12, color=COHORT_COLORS[cohort], label=cohort if x == 0 else None)
            ax[1, col].hlines(d.d.median(), x + 0.3 * i - 0.12, x + 0.3 * i + 0.12, color='k')
        x += 1
    ax[1, col].axhline(0, color='grey', lw=0.8)
    ax[1, col].set(xticks=np.arange(3) + 0.15, xticklabels=list(contrasts_A),
                   ylabel='session median of within-neuron difference' + (' (log2 FR)' if metric.startswith('fr') else ''),
                   title=f'Part A: {metric} (θ = 0.8)')
    ax[1, col].legend()
fig.tight_layout()
fig.savefig(PROJECT / 'exploratory-analyses' / '002_ff_pilot.png', dpi=110)

# text summary: Part A medians per theta and cohort
for theta in THETAS:
    a = df[df.theta == theta].pivot_table(index=['session_id', 'recording_id', 'unit_index', 'cohort'],
                                          columns='condition', values=METRICS)
    if a.empty:
        continue
    out = {}
    for label, (c1, c2) in contrasts_A.items():
        for metric in METRICS:
            out[(label, metric)] = (a[(metric, c1)] - a[(metric, c2)]).groupby(['session_id', 'cohort']).median() \
                .groupby('cohort').median()
    print(f'\nθ = {theta}: sessions per cohort',
          df[(df.theta == theta) & (df.condition == 'dis')].groupby('cohort').session_id.nunique().to_dict())
    print(pd.DataFrame(out).T.round(3))
