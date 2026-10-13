"""Reuse the mouse-level exploration/confirmation split from `aging_rt_variability` (user choice 2026-10-12).

The mouse is the independent unit, so every session of a mouse lands in the same set. Reusing the split keeps the
two projects comparable (H4 links engagement states to the RT-variability result) and keeps all GLM-HMM
exploration off the confirmation mice. Also plots what matters for GLM-HMM fitting: age, trials per mouse, and
the churchlandlab within-lab contrast.
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
OUT = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = OUT.parents[1] / 'aging-rt-variability' / 'code' / 'artifacts' / 'split.csv'
COLORS = {'exploration': '#1baf7a', 'confirmation': '#2a78d6'}

shutil.copy(SOURCE_SPLIT, OUT / 'artifacts' / 'split.csv')
split = pd.read_csv(OUT / 'artifacts' / 'split.csv')
sessions = pd.read_parquet(DATA / 'sessions.parquet')
mice = split.merge(sessions.groupby('subject_id').n_trials.sum().rename('n_trials').reset_index(), on='subject_id')
mice['churchland'] = mice.lab == 'churchlandlab'

print(mice.groupby(['split', 'cohort']).agg(mice=('subject_id', 'size'), sessions=('n_sessions', 'sum'),
                                             trials_median=('n_trials', 'median'), trials_min=('n_trials', 'min'),
                                             age_min=('mean_age_days', 'min'), age_max=('mean_age_days', 'max')))
print('churchlandlab only:')
print(mice[mice.churchland].groupby(['split', 'cohort']).agg(mice=('subject_id', 'size'),
                                                             trials_median=('n_trials', 'median')))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
# Left: trials per mouse vs age. Fewer trials (lifespan, 1-3 sessions) means noisier per-mouse state estimates.
for s, c in COLORS.items():
    for cohort, marker in [('bwm', 'o'), ('lifespan', '^')]:
        m = mice[(mice.split == s) & (mice.cohort == cohort)]
        ax[0].scatter(m.mean_age_days, m.n_trials, s=36, marker=marker, color=c, edgecolor='white', lw=0.8,
                      label=f'{s}, {cohort} (n={len(m)})')
ax[0].set(xlabel='mean age at session (days)', ylabel='trials per mouse (all sessions)', yscale='log',
          title='Trials per mouse vs age')
ax[0].legend(fontsize=8, frameon=False)
# Right: the churchlandlab within-lab contrast, the cleanest age comparison
counts = (mice[mice.churchland].groupby(['cohort', 'split']).size().unstack()[['exploration', 'confirmation']])
counts.plot.barh(ax=ax[1], color=[COLORS['exploration'], COLORS['confirmation']], width=0.7)
ax[1].set(xlabel='mice', ylabel='', title='churchlandlab mice by cohort and set')
ax[1].legend(frameon=False)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '000_data_split.png', dpi=120)
