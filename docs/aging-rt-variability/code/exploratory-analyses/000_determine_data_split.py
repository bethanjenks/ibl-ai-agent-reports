"""Split mice into exploration (30%) and confirmation (70%) sets.

The mouse is the independent unit, so the split is by mouse: every session of a mouse lands in the same set.
BWM mice are stratified by lab (lab is a known source of behavioural differences). Lifespan mice are almost all
from one lab, so they are split as one stratum. Within each stratum mice are sorted by age and sampled at
evenly spaced ranks, so both sets span the same age range.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
OUT = Path(__file__).resolve().parents[1]
EXPLORE_FRACTION = 0.3
SEED = 20261008
rng = np.random.default_rng(SEED)

sessions = pd.read_parquet(DATA / 'sessions.parquet')
mice = (sessions.groupby('subject_id')
        .agg(cohort=('source_group', 'first'), lab=('lab', 'first'),
             mean_age_days=('age_at_session_days', 'mean'), n_sessions=('session_id', 'size'))
        .reset_index())
mice['stratum'] = np.where(mice.cohort == 'bwm', 'bwm_' + mice.lab, 'lifespan')


def assign(stratum):
    """Age-stratified systematic sampling within one stratum.
    Mice are sorted by age and round(EXPLORE_FRACTION * n) of them are taken at evenly spaced ranks with a random
    offset, so the exploration mice cover the stratum's age range.
    Returns a boolean Series (True = exploration) indexed like `stratum`."""
    order = stratum.sort_values('mean_age_days').index
    n_explore = int(round(EXPLORE_FRACTION * len(order)))
    ranks = np.floor((np.arange(n_explore) + rng.random()) * len(order) / n_explore).astype(int)
    return pd.Series(stratum.index.isin(order[ranks]), index=stratum.index)


mice['split'] = np.where(pd.concat([assign(g) for _, g in mice.groupby('stratum')]).loc[mice.index],
                         'exploration', 'confirmation')
mice.drop(columns='stratum').to_csv(OUT / 'artifacts' / 'split.csv', index=False)

summary = mice.groupby(['cohort', 'split']).agg(mice=('subject_id', 'size'), sessions=('n_sessions', 'sum'),
                                               age_median=('mean_age_days', 'median'),
                                               age_min=('mean_age_days', 'min'), age_max=('mean_age_days', 'max'))
print(summary)
print(pd.crosstab(mice.lab, [mice.cohort, mice.split]))

# Balance figure: age distribution per set and cohort, and mice per lab
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
bins = np.arange(80, 620, 20)
for split, color in [('exploration', 'C1'), ('confirmation', 'C0')]:
    for cohort, hatch in [('bwm', None), ('lifespan', '//')]:
        m = mice[(mice.split == split) & (mice.cohort == cohort)]
        ax[0].hist(m.mean_age_days, bins, histtype='step', color=color, hatch=hatch, lw=1.5,
                   label=f'{split} {cohort} (n={len(m)})')
ax[0].set(xlabel='mean age at session (days)', ylabel='mice', title='Age per mouse by set and cohort')
ax[0].legend(fontsize=8)
counts = pd.crosstab(mice.lab, mice.split)[['exploration', 'confirmation']]
counts.plot.barh(ax=ax[1], color=['C1', 'C0'])
ax[1].set(xlabel='mice', ylabel='', title='Mice per lab (both cohorts)')
fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '000_determine_data_split.png', dpi=120)
