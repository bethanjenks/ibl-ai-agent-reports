"""Define and validate RT location/spread metrics on exploration sessions (primary trial rules).

Figure rows:
1. Didactic: one session's RTs and how each metric is read off them (linear scale, log scale, per contrast).
2. Diagnostics: mean–SD coupling, pooled vs per-contrast CV, agreement between spread metrics.
3. Reliability: odd/even trial split within session, and session-to-session within mouse. A metric that is not
   reliable within a mouse cannot track a between-mouse factor like age.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from rt_metrics import session_metrics, spread_metrics

OUT = Path(__file__).resolve().parents[1]
COHORT_COLOR = {'bwm': 'C0', 'lifespan': 'C3'}
METRICS = ['median_rt', 'cv', 'robust_cv', 'sd_log_rt', 'pc_cv', 'pc_robust_cv', 'pc_sd_log_rt', 'frac_fast', 'frac_slow']

trials = pd.read_parquet(OUT / 'artifacts' / '001_trials_rt_exploration.parquet')
sess = session_metrics(trials)
sess.to_csv(OUT / 'artifacts' / '002_session_metrics_primary.csv', index=False)
print(sess[METRICS].describe().round(3).T)

kept = trials[trials.paper_keep]
example = sess.iloc[(sess.cv - sess.cv.median()).abs().argmin()].session_id  # a typical session
rt = kept[kept.session_id == example].rt.values
m = spread_metrics(rt)

fig, ax = plt.subplots(3, 3, figsize=(16, 13))

# Row 1: didactic
a = ax[0, 0]
a.hist(rt, np.linspace(0, 2, 60), color='0.7')
q25, q75 = np.quantile(rt, [0.25, 0.75])
a.axvline(m['median_rt'], color='k', label=f"median = {m['median_rt']:.2f} s")
a.axvspan(q25, q75, color='C1', alpha=0.3, label=f"IQR → robust CV = IQR/median = {m['robust_cv']:.2f}")
a.errorbar(rt.mean(), a.get_ylim()[1] * 0.9, xerr=rt.std(ddof=1), fmt='o', color='C0', capsize=4,
           label=f"mean ± SD → CV = SD/mean = {m['cv']:.2f}")
a.set(xlabel='RT (s)', ylabel='trials', title=f'Example session (n={len(rt)} kept trials), linear scale')
a.legend(fontsize=8)

a = ax[0, 1]
lr = np.log(rt)
a.hist(lr, 50, color='0.7')
a.errorbar(lr.mean(), a.get_ylim()[1] * 0.9, xerr=lr.std(ddof=1), fmt='o', color='C2', capsize=4,
           label=f"SD of log RT = {m['sd_log_rt']:.2f}")
a.set(xlabel='log RT (log s)', ylabel='trials', title='Same session, log scale')
a.legend(fontsize=8)

a = ax[0, 2]
ex = kept[kept.session_id == example]
cs = sorted(ex.contrast.unique())
a.boxplot([ex.rt[ex.contrast == c] for c in cs], tick_labels=[f'{c:g}' for c in cs], showfliers=False)
for i, c in enumerate(cs):
    r = ex.rt[ex.contrast == c].values
    a.text(i + 1, 2.05, f'CV {r.std(ddof=1) / r.mean():.2f}\nn={len(r)}', ha='center', fontsize=7)
a.set(ylim=(0, 2.3), xlabel='unsigned contrast', ylabel='RT (s)',
      title=f"Per contrast; pc_CV = mean of per-contrast CVs = {sess.set_index('session_id').loc[example, 'pc_cv']:.2f}")

# Row 2: diagnostics
a = ax[1, 0]
sdv = kept.groupby('session_id').rt.agg(['mean', 'std']).join(sess.set_index('session_id').cohort)
for c, g in sdv.groupby('cohort'):
    a.scatter(g['mean'], g['std'], s=12, color=COHORT_COLOR[c], label=c)
r, p = stats.spearmanr(sdv['mean'], sdv['std'])
a.set(xlabel='session mean RT (s)', ylabel='session SD RT (s)', title=f'Mean–SD coupling (Spearman ρ={r:.2f})')
a.legend(fontsize=8)

METRIC_COLOR = {'cv': 'C0', 'robust_cv': 'C1', 'sd_log_rt': 'C2'}
a = ax[1, 1]
for metric, mk in [('cv', 'o'), ('sd_log_rt', 's'), ('robust_cv', '^')]:
    r, _ = stats.spearmanr(sess.median_rt, sess[metric])
    a.scatter(sess.median_rt, sess[metric], s=10, marker=mk, color=METRIC_COLOR[metric], label=f'{metric} (ρ={r:.2f})')
a.set(xscale='log', xlabel='session median RT (s)', ylabel='spread metric', title='Spread vs median RT')
a.legend(fontsize=8)

a = ax[1, 2]
pairs = [('cv', 'pc_cv'), ('robust_cv', 'pc_robust_cv'), ('sd_log_rt', 'pc_sd_log_rt')]
for x, y in pairs:
    r, _ = stats.spearmanr(sess[x], sess[y])
    a.scatter(sess[x], sess[y], s=10, color=METRIC_COLOR[x], label=f'{x} vs per-contrast (ρ={r:.2f})')
lim = [0, max(sess[[*sum(pairs, ())]].max())]
a.plot(lim, lim, 'k', lw=0.6)
a.set(xlabel='pooled', ylabel='per-contrast average', title='Pooled vs per-contrast spread')
a.legend(fontsize=8)

# Row 3: reliability
odd = session_metrics(trials[trials.trial_id % 2 == 1].assign(trial_id=lambda d: d.trial_id // 2))
even = session_metrics(trials[trials.trial_id % 2 == 0].assign(trial_id=lambda d: d.trial_id // 2))
split_half = {k: stats.spearmanr(odd[k], even[k], nan_policy='omit')[0] for k in METRICS}

# Session-to-session within mouse: first vs second session of each mouse with ≥ 2 sessions
s2 = sess.sort_values(['subject_id', 'age_at_session_days']).groupby('subject_id').filter(lambda g: len(g) >= 2)
first_two = s2.groupby('subject_id').head(2).assign(k=lambda d: d.groupby('subject_id').cumcount())
wide = first_two.pivot(index='subject_id', columns='k', values=METRICS)
between = {k: stats.spearmanr(wide[(k, 0)], wide[(k, 1)], nan_policy='omit')[0] for k in METRICS}

a = ax[2, 0]
x = np.arange(len(METRICS))
a.bar(x - 0.2, [split_half[k] for k in METRICS], 0.4, label='odd vs even trials (same session)')
a.bar(x + 0.2, [between[k] for k in METRICS], 0.4, label=f'session 1 vs 2, same mouse (n={len(wide)} mice)')
a.set_xticks(x, METRICS, rotation=45, ha='right')
a.set(ylabel='Spearman ρ', title='Reliability (exploration, primary trial rules)', ylim=(0, 1))
a.legend(fontsize=8)

for a, k in zip(ax[2, 1:], ['cv', 'sd_log_rt']):
    for c in ['bwm', 'lifespan']:
        mice = first_two[first_two.cohort == c].subject_id.unique()
        a.scatter(wide.loc[wide.index.isin(mice), (k, 0)], wide.loc[wide.index.isin(mice), (k, 1)], s=15,
                  color=COHORT_COLOR[c], label=c)
    a.set(xlabel=f'{k}, session 1', ylabel=f'{k}, session 2', title=f'{k}: within-mouse session-to-session (ρ={between[k]:.2f})')
    a.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '002_variability_metrics.png', dpi=110)
print(pd.DataFrame({'split_half': split_half, 'session1_vs_2': between}).round(2))
