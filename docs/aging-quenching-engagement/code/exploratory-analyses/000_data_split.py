"""Reuse the aging_glmhmm_engagement mouse split (user, 2026-10-13) and check what the FF analysis can use.

The engagement posteriors were fitted separately on exploration and confirmation mice, so copying that split keeps
confirmation labels untouched. This script counts, for exploration mice only:
- neurons per paper ROI and cohort (FF needs enough neurons in each ROI in both cohorts);
- engaged (P > 0.8) and disengaged (P < 0.2) trials per session after the paper's trial filters (a disengaged-trial
  FF needs enough trials);
- P(engaged) against trial position, because if disengagement sits late in sessions, time-in-session drift can mimic
  a state effect.
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quench_common import PROJECT, GLMHMM, load_trials, load_units

THETA = 0.8
COHORT_COLORS = {'bwm': '#2a78d6', 'lifespan': '#d6552a'}

shutil.copy(GLMHMM / 'split.csv', PROJECT / 'artifacts' / 'split.csv')
units = load_units()
units = units[units.split == 'exploration']
trials = load_trials().merge(units[['session_id', 'subject_id', 'cohort']].drop_duplicates(), on='session_id')

neurons = units.groupby(['roi', 'cohort']).agg(neurons=('unit_index', 'size'), mice=('subject_id', 'nunique'))
neurons = neurons.unstack('cohort').fillna(0).astype(int)
print('exploration neurons (mice) per ROI:\n', neurons)

trials['state'] = np.select([trials.p_engaged > THETA, trials.p_engaged < 1 - THETA], ['engaged', 'disengaged'],
                            'uncertain')
per_session = trials.groupby(['cohort', 'session_id']).state.value_counts().unstack(fill_value=0)
print('\ntrials per session (paper filters) by state:\n', per_session.groupby('cohort').median())
for n_min in [20, 50, 100]:
    print(f'sessions with >= {n_min} disengaged trials:',
          (per_session.disengaged >= n_min).groupby('cohort').agg(['sum', 'size']).to_dict('index'))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.arange(len(neurons))
for i, cohort in enumerate(['bwm', 'lifespan']):
    ax[0].bar(x + 0.4 * i, neurons['neurons'][cohort], 0.4, color=COHORT_COLORS[cohort], label=cohort)
ax[0].set(xticks=x + 0.2, yscale='log', ylabel='neurons (exploration)', title='Neurons per paper ROI')
ax[0].set_xticklabels(neurons.index, rotation=90)
ax[0].legend()
for cohort, d in per_session.groupby('cohort'):
    ax[1].scatter(d.engaged, d.disengaged + 1, s=12, color=COHORT_COLORS[cohort], label=cohort)
ax[1].set(xlabel=f'engaged trials (P > {THETA})', ylabel=f'disengaged trials + 1 (P < {1 - THETA:.1f})',
          yscale='log', title='Trials per session after paper filters')
ax[1].legend()
trials['position'] = trials.trial_rank // 100 * 100
for cohort, d in trials.groupby('cohort'):
    m = d.groupby('position').p_engaged.agg(['mean', 'size'])
    m = m[m['size'] >= 200]
    ax[2].plot(m.index, m['mean'], 'o-', color=COHORT_COLORS[cohort], label=cohort)
ax[2].set(xlabel='trial position in session (100-trial bins)', ylabel='mean P(engaged)', ylim=(0, 1),
          title='Engagement vs time in session')
ax[2].axvline(400, color='grey', ls='--', lw=1)
ax[2].legend()
fig.tight_layout()
fig.savefig(PROJECT / 'exploratory-analyses' / '000_data_split.png', dpi=120)
