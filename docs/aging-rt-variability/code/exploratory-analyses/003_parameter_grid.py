"""How the age association of each RT metric depends on trial-selection parameters (exploration mice only).

Effect size = Spearman ρ between mouse-level metric (mean over sessions) and mouse mean age, for all exploration
mice (48) and BWM mice only (42; removes the cohort contrast). Rank correlation, so a few extreme mice cannot
dominate. This is a sensitivity map, not a test: the point is whether sign and size are stable across settings.
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from rt_metrics import session_metrics, mouse_metrics

OUT = Path(__file__).resolve().parents[1]
LOWER = [0.0, 0.08, 0.15]      # s; 0.0 keeps all positive RTs
UPPER = [1.0, 2.0, 5.0, 60.0]  # s; 60 = trial timeout, i.e. no upper cut
DROP_EARLY = [False, True]
FIRST_N = [400, 10_000]        # 10_000 = all trials
FAST = [0.08, 0.1, 0.15]       # anticipation thresholds for frac_fast
SLOW = [1.0, 2.0, 5.0]         # slow-trial thresholds for frac_slow
SPREAD = ['median_rt', 'cv', 'pc_cv', 'sd_log_rt', 'robust_cv']
PRIMARY = dict(lower=0.08, upper=2.0, drop_early=False, first_n=400)

trials = pd.read_parquet(OUT / 'artifacts' / '001_trials_rt_exploration.parquet')
trials = trials[trials.rt > 0]  # 8 trials have response before stimOn; no RT can be defined


def age_rho(sessions, metric):
    """Spearman ρ of mouse-level metric vs mouse mean age: all mice, BWM mice only."""
    m = mouse_metrics(sessions)
    b = m[m.cohort == 'bwm']
    return (stats.spearmanr(m.age_at_session_days, m[metric])[0],
            stats.spearmanr(b.age_at_session_days, b[metric])[0])


rows, all_sessions = [], []
for lo, hi, de, fn in itertools.product(LOWER, UPPER, DROP_EARLY, FIRST_N):
    s = session_metrics(trials, window=(lo, hi), drop_early=de, first_n=fn)
    all_sessions.append(s.assign(lower=lo, upper=hi, drop_early=de, first_n=fn))
    for metric in SPREAD:
        r_all, r_bwm = age_rho(s, metric)
        rows.append(dict(lower=lo, upper=hi, drop_early=de, first_n=fn, metric=metric, rho_all=r_all, rho_bwm=r_bwm))
grid = pd.DataFrame(rows)
pd.concat(all_sessions).to_parquet(OUT / 'artifacts' / '003_session_metrics_grid.parquet')

tail_rows = []
for fast, slow, fn in itertools.product(FAST, SLOW, FIRST_N):
    s = session_metrics(trials, first_n=fn, fast=fast, slow=slow)
    for metric, thr in [('frac_fast', fast), ('frac_slow', slow)]:
        r_all, r_bwm = age_rho(s, metric)
        tail_rows.append(dict(metric=metric, threshold=thr, first_n=fn, rho_all=r_all, rho_bwm=r_bwm))
tails = pd.DataFrame(tail_rows).drop_duplicates(['metric', 'threshold', 'first_n'])
grid.to_csv(OUT / 'artifacts' / '003_grid_age_rho.csv', index=False)
tails.to_csv(OUT / 'artifacts' / '003_tail_age_rho.csv', index=False)

# Bootstrap over mice at the primary setting, to show how noisy ρ is with 48 mice
rng = np.random.default_rng(0)
prim = mouse_metrics(session_metrics(trials, window=(PRIMARY['lower'], PRIMARY['upper'])))
boot = {k: np.percentile([stats.spearmanr(*prim.sample(len(prim), replace=True, random_state=rng)[['age_at_session_days', k]].values.T)[0]
                          for _ in range(2000)], [2.5, 97.5]) for k in SPREAD}
print('primary setting, all mice: ρ and 95% bootstrap CI')
for k in SPREAD:
    print(f"  {k:10s} ρ={stats.spearmanr(prim.age_at_session_days, prim[k])[0]:+.2f}  CI [{boot[k][0]:+.2f}, {boot[k][1]:+.2f}]")
print(grid.groupby('metric')[['rho_all', 'rho_bwm']].describe().round(2).T)
print(tails.round(2))

# Figure: one heatmap per metric and mouse set; rows = RT window, columns = early-movement handling × trial count
fig, ax = plt.subplots(3, len(SPREAD), figsize=(24, 14), gridspec_kw={'height_ratios': [1, 1, 0.8]})
for j, metric in enumerate(SPREAD):
    for i, col in enumerate(['rho_all', 'rho_bwm']):
        g = grid[grid.metric == metric].pivot_table(index=['lower', 'upper'], columns=['drop_early', 'first_n'], values=col)
        a = ax[i, j]
        im = a.imshow(g.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
        for (r, c), v in np.ndenumerate(g.values):
            star = '*' if g.index[r] == (PRIMARY['lower'], PRIMARY['upper']) and g.columns[c] == (False, 400) else ''
            a.text(c, r, f'{v:+.2f}{star}', ha='center', va='center', fontsize=8)
        a.set_yticks(range(len(g)), [f'{lo:g}–{hi:g} s' for lo, hi in g.index] if j == 0 else [])
        if j == 0:
            a.set_ylabel('RT window')
        a.set_xticks(range(g.shape[1]), [f"{'drop' if de else 'keep'} early\n{'first 400' if fn == 400 else 'all'}" for de, fn in g.columns], fontsize=7)
        a.set_title(f"{metric}: ρ with age, {'all 48 mice' if col == 'rho_all' else 'BWM 42 mice'}", fontsize=10)
fig.colorbar(im, ax=ax[:2, :], shrink=0.5, label='Spearman ρ (mouse level)')
for j, metric in enumerate(['frac_fast', 'frac_slow']):
    a = ax[2, j]
    t = tails[tails.metric == metric]
    for (col, fn), mk in zip(itertools.product(['rho_all', 'rho_bwm'], FIRST_N), ['o-', 's-', 'o--', 's--']):
        tt = t[t.first_n == fn]
        a.plot(tt.threshold, tt[col], mk, label=f"{'all' if col == 'rho_all' else 'BWM'}, {'first 400' if fn == 400 else 'all trials'}")
    a.axhline(0, color='k', lw=0.6)
    a.set(xlabel='threshold (s)', ylabel='Spearman ρ with age', ylim=(-0.6, 0.6),
          title=f"{metric} (RT {'<' if metric == 'frac_fast' else '>'} threshold)")
    a.legend(fontsize=8)
a = ax[2, 2]
x = np.arange(len(SPREAD))
r = [stats.spearmanr(prim.age_at_session_days, prim[k])[0] for k in SPREAD]
a.errorbar(x, r, yerr=np.abs(np.array([boot[k] for k in SPREAD]).T - r), fmt='o', capsize=4)
a.axhline(0, color='k', lw=0.6)
a.set_xticks(x, SPREAD, rotation=30, ha='right')
a.set(ylabel='ρ with age (95% bootstrap CI)', title='Primary setting, all 48 mice', ylim=(-0.8, 0.8))
for a in ax[2, 3:]:
    a.axis('off')
fig.savefig(OUT / 'exploratory-analyses' / '003_parameter_grid.png', dpi=100, bbox_inches='tight')
