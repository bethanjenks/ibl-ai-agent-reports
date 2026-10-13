"""Build per-trial RT table for exploration mice and draw metric-validation diagnostics.

RT (primary, the paper's definition) = response_times - stimOn_times: time from stimulus onset until the wheel
reaches the response threshold. Movement-onset RT = firstMovement_times - stimOn_times is a secondary check.
The paper's trial rules are flagged, not applied, so their effect can be inspected:
first 400 trials; required events present; RT in [0.08, 2] s.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025')
OUT = Path(__file__).resolve().parents[1]
RT_WINDOW = (0.08, 2.0)  # s, paper's window
N_FIRST = 400            # paper keeps the first 400 trials per session
REQUIRED = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times', 'stimOn_times', 'firstMovement_times']
COHORT_COLOR = {'bwm': 'C0', 'lifespan': 'C3'}

split = pd.read_csv(OUT / 'artifacts' / 'split.csv')
sessions = pd.read_parquet(DATA / 'metadata' / 'sessions.parquet')
sessions = sessions.merge(split[['subject_id', 'split']], on='subject_id')
sessions = sessions[sessions.split == 'exploration']
trials = pd.read_parquet(DATA / 'metadata' / 'trials.parquet')
trials = trials.merge(sessions[['session_id', 'subject_id', 'source_group', 'lab', 'age_at_session_days']],
                      on='session_id').rename(columns={'source_group': 'cohort'})

trials['rt'] = trials.response_times - trials.stimOn_times
trials['movement_rt'] = trials.firstMovement_times - trials.stimOn_times
trials['contrast'] = trials.contrastLeft.fillna(0) + trials.contrastRight.fillna(0)  # unsigned contrast
trials['first_n'] = trials.trial_id < N_FIRST
trials['complete'] = trials[REQUIRED].notna().all(axis=1) & (trials.choice != 0)
trials['in_window'] = trials.rt.between(*RT_WINDOW)
trials['early_move'] = trials.movement_rt < 0  # wheel moved before the stimulus appeared
trials['paper_keep'] = trials.first_n & trials.complete & trials.in_window
trials.to_parquet(OUT / 'artifacts' / '001_trials_rt_exploration.parquet')

# How many trials each rule removes, by cohort (rules applied in order)
first = trials[trials.first_n]
comp = first[first.complete]
counts = pd.DataFrame({
    'trials': trials.groupby('cohort').size(),
    'first_400': first.groupby('cohort').size(),
    'complete': comp.groupby('cohort').size(),
    'rt<0.08': comp[comp.rt < RT_WINDOW[0]].groupby('cohort').size(),
    'rt>2': comp[comp.rt > RT_WINDOW[1]].groupby('cohort').size(),
    'kept': comp[comp.in_window].groupby('cohort').size(),
    'early_move_in_kept': comp[comp.in_window & comp.early_move].groupby('cohort').size()})
print(counts)
print((counts[['rt<0.08', 'rt>2', 'early_move_in_kept']].div(counts.complete, axis=0) * 100).round(1), '(% of complete)')

per_session = (comp.groupby(['session_id', 'cohort', 'age_at_session_days'])
               .agg(frac_slow=('rt', lambda r: np.mean(r > RT_WINDOW[1])),
                    frac_fast=('rt', lambda r: np.mean(r < RT_WINDOW[0])),
                    frac_early=('early_move', 'mean'), median_rt=('rt', 'median'))
               .reset_index())

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 3)

# A, B: raw wheel position around stimulus onset for example trials, with the two RT anchors marked
for col, cohort in enumerate(['bwm', 'lifespan']):
    ax = fig.add_subplot(gs[0, col])
    sid = per_session[per_session.cohort == cohort].sort_values('median_rt').session_id.iloc[len(per_session[per_session.cohort == cohort]) // 2]
    wheel = pd.read_parquet(DATA / 'wheel' / f'{sid}.parquet')
    ex = comp[comp.session_id == sid].iloc[100:112]
    for i, tr in enumerate(ex.itertuples()):
        m = (wheel.time > tr.stimOn_times - 0.5) & (wheel.time < tr.stimOn_times + 3)
        t = wheel.time[m] - tr.stimOn_times
        pos = wheel.position[m] - np.interp(tr.stimOn_times, wheel.time, wheel.position)
        ax.plot(t, pos + 0.6 * i, color='k', lw=0.8)
        ax.plot(tr.movement_rt, np.interp(tr.firstMovement_times, wheel.time, wheel.position) - np.interp(tr.stimOn_times, wheel.time, wheel.position) + 0.6 * i, 'o', color='C2', ms=5)
        ax.plot(tr.rt, np.interp(tr.response_times, wheel.time, wheel.position) - np.interp(tr.stimOn_times, wheel.time, wheel.position) + 0.6 * i, 'v', color='C1', ms=6)
        ax.text(3.02, 0.6 * i, f'c={tr.contrast:g}', fontsize=7, va='center')
    ax.axvline(0, color='grey', ls='--')
    ax.axvspan(RT_WINDOW[1], 3, color='grey', alpha=0.15)
    ax.set(xlabel='time from stimOn (s)', ylabel='wheel position (rad, offset per trial)',
           title=f'{cohort} example session, trials 100-111\n● first movement  ▼ response (RT end); grey = RT > 2 s')

# C: RT distributions per contrast, log-binned, by cohort
ax = fig.add_subplot(gs[0, 2])
bins = np.logspace(np.log10(0.02), np.log10(60), 60)
for cohort, ls in [('bwm', '-'), ('lifespan', '--')]:
    for c, color in zip([0, 0.0625, 0.25, 1.0], plt.cm.viridis(np.linspace(0, 0.85, 4))):
        r = comp[(comp.cohort == cohort) & np.isclose(comp.contrast, c)].rt
        ax.hist(r, bins, density=True, histtype='step', ls=ls, color=color, label=f'{cohort} c={c:g}')
for x in RT_WINDOW:
    ax.axvline(x, color='k', lw=0.8)
ax.set(xscale='log', xlabel='RT = response − stimOn (s)', ylabel='density', title='RT distributions (first 400, complete)')
ax.legend(fontsize=7, ncol=2)

# D: response RT vs movement-onset RT
ax = fig.add_subplot(gs[1, 0])
ok = (comp.rt > 0) & (comp.movement_rt > 0)
ax.hist2d(np.log10(comp.movement_rt[ok]), np.log10(comp.rt[ok]), bins=100, cmap='Greys', norm='log')
ax.plot([-2, 2], [-2, 2], 'r', lw=0.7)
ax.set(xlabel='log10 movement-onset RT (s)', ylabel='log10 response RT (s)',
       title=f'Response vs movement-onset RT\n(excludes {np.mean(~ok):.1%} trials with movement before stimOn)')

# E: fraction of trials cut by the window, per session vs age
ax = fig.add_subplot(gs[1, 1])
for cohort, g in per_session.groupby('cohort'):
    ax.scatter(g.age_at_session_days, g.frac_slow * 100, color=COHORT_COLOR[cohort], s=15, label=f'{cohort}: RT > 2 s')
    ax.scatter(g.age_at_session_days, g.frac_fast * 100, color=COHORT_COLOR[cohort], s=15, marker='x', label=f'{cohort}: RT < 0.08 s')
ax.set(xlabel='age at session (days)', ylabel='% of complete trials', title='Trials removed by the 0.08–2 s window, per session')
ax.legend(fontsize=8)

# F: early movements per session vs age
ax = fig.add_subplot(gs[1, 2])
for cohort, g in per_session.groupby('cohort'):
    ax.scatter(g.age_at_session_days, g.frac_early * 100, color=COHORT_COLOR[cohort], s=15, label=cohort)
ax.set(xlabel='age at session (days)', ylabel='% of complete trials',
       title='Wheel movement before stimOn (firstMovement < stimOn)')
ax.legend(fontsize=8)

# G: RT across the session (time-in-session effect; first-400 cutoff marked)
ax = fig.add_subplot(gs[2, 0:2])
ok = trials.complete & (trials.rt > 0)
tb = (trials.trial_id[ok] // 50) * 50
for cohort in ['bwm', 'lifespan']:
    sub = trials[ok & (trials.cohort == cohort)]
    q = sub.groupby(tb[sub.index]).rt.quantile([0.25, 0.5, 0.75]).unstack()
    n_sess = sub.groupby(tb[sub.index]).session_id.nunique()
    q = q[n_sess >= 5]
    ax.plot(q.index + 25, q[0.5], color=COHORT_COLOR[cohort], label=f'{cohort} median')
    ax.fill_between(q.index + 25, q[0.25], q[0.75], color=COHORT_COLOR[cohort], alpha=0.2, label=f'{cohort} IQR')
ax.axvline(N_FIRST, color='k', ls='--', label='first 400 cutoff')
ax.set(yscale='log', xlabel='trial number (bins of 50; bins with ≥ 5 sessions)', ylabel='RT (s)',
       title='RT across the session, all complete trials')
ax.legend(fontsize=8)

# H: session median RT vs age (preview only; not a test)
ax = fig.add_subplot(gs[2, 2])
for cohort, g in per_session.groupby('cohort'):
    ax.scatter(g.age_at_session_days, g.median_rt, color=COHORT_COLOR[cohort], s=15, label=cohort)
ax.set(yscale='log', xlabel='age at session (days)', ylabel='median RT, unwindowed (s)', title='Session median RT (first 400, complete)')
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '001_rt_definitions.png', dpi=110)
