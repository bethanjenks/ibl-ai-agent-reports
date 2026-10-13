"""Post-hoc descriptives on the confirmation set (declared 'descriptive only' in the locked plan; no tests).

1. Window sensitivity: mouse-level ρ(age, metric) across RT windows, for sd_log_rt and CV, in exploration and
   confirmation mice. Does the CV sign flip replicate?
2. Median RT vs age across both cohorts.
3. Tail rates (RT < 0.15 s, RT > 2 s) vs age.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / 'exploratory-analyses'))
from rt_metrics import session_metrics, mouse_metrics

WINDOWS = [(0.08, 1.0), (0.08, 2.0), (0.08, 5.0), (0.08, 60.0)]
COHORT_COLOR = {'bwm': 'C0', 'lifespan': 'C3'}
sets = {'exploration': pd.read_parquet(OUT / 'artifacts' / '001_trials_rt_exploration.parquet'),
        'confirmation': pd.read_parquet(OUT / 'artifacts' / '005_trials_rt_confirmation.parquet')}

rows = []
for name, tr in sets.items():
    for w in WINDOWS:
        m = mouse_metrics(session_metrics(tr, window=w))
        for metric in ['sd_log_rt', 'cv']:
            rows.append(dict(set=name, upper=w[1], metric=metric, rho=stats.spearmanr(m.age_at_session_days, m[metric])[0]))
sens = pd.DataFrame(rows)
sens.to_csv(OUT / 'confirmatory-analyses' / '006_window_sensitivity.csv', index=False)
print(sens.pivot_table(index=['metric', 'upper'], columns='set', values='rho').round(2))

conf = mouse_metrics(session_metrics(sets['confirmation'], fast=0.15, slow=2.0))
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
a = ax[0]
for (name, metric), g in sens.groupby(['set', 'metric']):
    a.plot(range(len(WINDOWS)), g.rho, 'o-' if name == 'confirmation' else 's--',
           color='C2' if metric == 'sd_log_rt' else 'C4', label=f'{metric}, {name}')
a.set_xticks(range(len(WINDOWS)), [f'0.08–{hi:g} s' for _, hi in WINDOWS])
a.axhline(0, color='k', lw=0.6)
a.set(xlabel='RT window', ylabel='Spearman ρ with age (mouse level)', title='Window sensitivity')
a.legend(fontsize=8)
for a, metric, label in zip(ax[1:], ['median_rt', 'frac_fast', 'frac_slow'], ['median RT (s)', 'fraction RT < 0.15 s', 'fraction RT > 2 s']):
    for c, g in conf.groupby('cohort'):
        a.scatter(g.age_at_session_days, g[metric], color=COHORT_COLOR[c], s=18, label=f'{c} (n={len(g)})')
    r = stats.spearmanr(conf.age_at_session_days, conf[metric])[0]
    rb = stats.spearmanr(conf[conf.cohort == 'bwm'].age_at_session_days, conf[conf.cohort == 'bwm'][metric])[0]
    a.set(xlabel='mouse mean age (days)', ylabel=label, title=f'Confirmation: ρ all = {r:+.2f}, BWM = {rb:+.2f}')
    a.legend(fontsize=8)
    print(metric, round(r, 2), round(rb, 2))
fig.tight_layout()
fig.savefig(OUT / 'confirmatory-analyses' / '006_posthoc_descriptives.png', dpi=110)
