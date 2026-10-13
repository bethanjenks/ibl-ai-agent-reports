"""Age effects on RT metrics in exploration mice, with adversarial checks (primary trial rules).

1. Mouse-level metric vs age, by cohort, raw and lab-centred (lab mean removed within BWM).
2. Session-level mixed models: metric_z ~ age_z [+ covariates] + (1 | mouse), mouse as random intercept so
   sessions are not treated as independent. Covariates test alternative explanations of an age effect:
   lab, cohort, session performance (accuracy on 100% contrast), number of trials completed, early-movement
   rate, and median RT (for spread metrics).
3. Time in session: spread metric from trials 0–199 vs 200–399.
4. Overlap window: BWM vs lifespan mice of similar age (300–460 days).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats
from rt_metrics import session_metrics, mouse_metrics

OUT = Path(__file__).resolve().parents[1]
METRICS = ['sd_log_rt', 'robust_cv', 'cv', 'median_rt']
COHORT_COLOR = {'bwm': 'C0', 'lifespan': 'C3'}
OVERLAP = (300, 460)  # days

trials = pd.read_parquet(OUT / 'artifacts' / '001_trials_rt_exploration.parquet')
trials = trials[trials.rt > 0]
sess = session_metrics(trials)
first = trials[(trials.trial_id < 400) & trials.complete]
cov = first.groupby('session_id').agg(acc_100=('feedbackType', lambda f: np.nan), frac_early=('early_move', 'mean'))
cov['acc_100'] = first[first.contrast == 1].groupby('session_id').feedbackType.apply(lambda f: np.mean(f == 1))
cov['n_trials_total'] = trials.groupby('session_id').trial_id.max() + 1
sess = sess.merge(cov.reset_index(), on='session_id')
sess['age_z'] = (sess.age_at_session_days - sess.age_at_session_days.mean()) / sess.age_at_session_days.std()
for c in METRICS + ['acc_100', 'frac_early', 'n_trials_total', 'median_rt']:
    sess[c + '_z'] = (sess[c] - sess[c].mean()) / sess[c].std()
mice = mouse_metrics(sess)

# 2. Mixed models
models = {'age only': '', '+ lab': ' + C(lab)', '+ cohort': ' + C(cohort)', '+ accuracy': ' + acc_100_z',
          '+ n trials': ' + n_trials_total_z', '+ early-move rate': ' + frac_early_z', '+ median RT': ' + median_rt_z'}
rows = []
for metric in METRICS:
    for name, extra in models.items():
        if name == '+ median RT' and metric == 'median_rt':
            continue
        fit = smf.mixedlm(f'{metric}_z ~ age_z{extra}', sess, groups=sess.subject_id).fit(reml=True)
        rows.append(dict(metric=metric, model=name, beta_age=fit.params['age_z'], ci_lo=fit.conf_int().loc['age_z', 0],
                         ci_hi=fit.conf_int().loc['age_z', 1], p=fit.pvalues['age_z']))
    b = sess[sess.cohort == 'bwm']
    fit = smf.mixedlm(f'{metric}_z ~ age_z', b, groups=b.subject_id).fit(reml=True)
    rows.append(dict(metric=metric, model='BWM only', beta_age=fit.params['age_z'], ci_lo=fit.conf_int().loc['age_z', 0],
                     ci_hi=fit.conf_int().loc['age_z', 1], p=fit.pvalues['age_z']))
mm = pd.DataFrame(rows)
mm.to_csv(OUT / 'artifacts' / '004_mixed_models_exploration.csv', index=False)
print(mm.round(3).to_string())

# 3. Time in session: first vs second half of the first 400 trials
halves = {h: session_metrics(trials[(trials.trial_id >= lo) & (trials.trial_id < lo + 200)].assign(trial_id=lambda d: d.trial_id - lo))
          for h, lo in [('trials 0–199', 0), ('trials 200–399', 200)]}

fig, ax = plt.subplots(3, 4, figsize=(20, 13))
for j, metric in enumerate(METRICS):
    a = ax[0, j]
    for c, g in mice.groupby('cohort'):
        a.scatter(g.age_at_session_days, g[metric], color=COHORT_COLOR[c], s=20, label=f'{c} (n={len(g)})')
    r_all = stats.spearmanr(mice.age_at_session_days, mice[metric])
    b = mice[mice.cohort == 'bwm']
    r_b = stats.spearmanr(b.age_at_session_days, b[metric])
    a.axvspan(*OVERLAP, color='0.9', zorder=-1)
    a.set(xlabel='mouse mean age (days)', ylabel=metric,
          title=f'{metric}\nall ρ={r_all[0]:+.2f}; BWM ρ={r_b[0]:+.2f}')
    a.legend(fontsize=8)

    # Lab-centred within BWM: does age explain variation between mice of the same lab?
    a = ax[1, j]
    b = b.assign(centred=b[metric] - b.groupby('lab')[metric].transform('mean'),
                 age_c=b.age_at_session_days - b.groupby('lab').age_at_session_days.transform('mean'))
    a.scatter(b.age_c, b.centred, s=20, c=pd.factorize(b.lab)[0], cmap='tab20')
    r = stats.spearmanr(b.age_c, b.centred)
    a.axhline(0, color='k', lw=0.5)
    a.set(xlabel='age − lab mean (days)', ylabel=f'{metric} − lab mean',
          title=f'BWM, lab-centred (colour = lab)\nρ={r[0]:+.2f}')

# Mixed-model forest plot
for j, metric in enumerate(METRICS[:2]):
    a = ax[2, j]
    d = mm[mm.metric == metric].reset_index(drop=True)
    a.errorbar(d.beta_age, d.index, xerr=[d.beta_age - d.ci_lo, d.ci_hi - d.beta_age], fmt='o', capsize=3)
    a.axvline(0, color='k', lw=0.6)
    a.set_yticks(d.index, d.model)
    a.invert_yaxis()
    a.set(xlabel='β age (SD of metric per SD of age), 95% CI', title=f'{metric}: mixed model, (1 | mouse)')

# Time in session
a = ax[2, 2]
for h, mk in zip(halves, ['o', 's']):
    m = mouse_metrics(halves[h])
    for c, g in m.groupby('cohort'):
        a.scatter(g.age_at_session_days, g.sd_log_rt, marker=mk, color=COHORT_COLOR[c], s=18, alpha=0.7,
                  label=f'{h}, {c}' if True else None)
    print(h, 'sd_log_rt ρ with age:', round(stats.spearmanr(m.age_at_session_days, m.sd_log_rt)[0], 2))
a.set(xlabel='mouse mean age (days)', ylabel='sd_log_rt', title='Time in session: first vs second 200 trials')
a.legend(fontsize=7)

# Overlap window: same ages, different cohort
a = ax[2, 3]
ov = sess[sess.age_at_session_days.between(*OVERLAP)]
ovm = mouse_metrics(ov)
for k, metric in enumerate(['sd_log_rt', 'robust_cv', 'cv']):
    for c, dx in [('bwm', -0.15), ('lifespan', 0.15)]:
        v = ovm[ovm.cohort == c][metric]
        a.scatter(np.full(len(v), k + dx), v, color=COHORT_COLOR[c], s=25, label=f'{c} (n={len(v)} mice)' if k == 0 else None)
a.set_xticks(range(3), ['sd_log_rt', 'robust_cv', 'cv'])
a.set(ylabel='metric', title=f'Overlap {OVERLAP[0]}–{OVERLAP[1]} d: cohort at equal age')
a.legend(fontsize=8)
print(ovm[['subject_id', 'cohort', 'age_at_session_days', 'sd_log_rt', 'robust_cv', 'cv']].round(2))

fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '004_age_effects_exploration.png', dpi=100)
