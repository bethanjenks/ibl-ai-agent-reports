"""POST HOC (user request 2026-10-13; not part of the locked plan). Engaged trials, equal N, as in 005.

(a) Within cohort: is there an age effect within bwm alone (92–461 d) and within lifespan alone? This asks whether
    the confirmed effects are a graded age effect or a cohort difference. Confirmation mice; mouse-permutation p,
    two-sided (post hoc), no multiplicity correction.
(b) Per ROI: β_age per paper ROI for quench and ff_post (paper windows), all 158 mice (exploration +
    confirmation pooled for coverage; descriptive), 95% mouse-bootstrap CI. ROIs need >= 3 lifespan and >= 3 bwm mice.
Outputs: 006_posthoc_within_cohort.csv, 006_posthoc_roi.csv, 006_posthoc_checks.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exploratory-analyses'))
from quench_common import PROJECT, METRICS, LOG_METRICS, age_slopes

N_PERM, N_BOOT, MIN_MICE = 5_000, 2_000, 3
rng = np.random.default_rng(7)
OUT = PROJECT / 'confirmatory-analyses'
load = lambda split: pd.read_parquet(PROJECT / 'artifacts' / f'005_partB_per_neuron_{split}.parquet')
conf, both = load('confirmation'), pd.concat([load('exploration'), load('confirmation')])


def usable(d, metric):
    return d[np.isfinite(d[metric]) & ((d[metric] > 0) | (metric not in LOG_METRICS))]


def mouse_ages(d):
    return d.groupby('subject_id').age_at_session_days.mean().sort_index().values / 100


def beta_perm(d, metric):
    """β_age (ROI fixed effects) and two-sided mouse-permutation p."""
    ages = mouse_ages(d)
    beta = age_slopes(d, metric, ages[None])[0]
    null = age_slopes(d, metric, np.stack([rng.permutation(ages) for _ in range(N_PERM)]))
    return beta, ((np.abs(null) >= abs(beta)).sum() + 1) / (N_PERM + 1)


def beta_boot(d, metric):
    """β_age and 95% mouse-bootstrap CI (single ROI, so no fixed effects needed beyond the intercept)."""
    y = np.log(d[metric].values) if metric in LOG_METRICS else d[metric].values
    mouse = np.unique(d.subject_id, return_inverse=True)[1]
    a = mouse_ages(d)[mouse]
    n_mice = mouse.max() + 1
    W = rng.multinomial(n_mice, np.full(n_mice, 1 / n_mice), size=N_BOOT)[:, mouse].astype(float)
    W = np.vstack([np.ones(len(y)), W])                           # row 0 = original data
    a_c = a - (W @ a / W.sum(1))[:, None]
    y_c = y - (W @ y / W.sum(1))[:, None]
    b = (W * a_c * y_c).sum(1) / (W * a_c ** 2).sum(1)
    return b[0], *np.nanpercentile(b[1:], [2.5, 97.5])


# (a) within cohort, confirmation mice
rows = []
eng = conf[conf.trial_set == 'eng']
for cohort in ['bwm', 'lifespan']:
    for variant in ['paper', 'safe']:
        for metric in METRICS:
            d = usable(eng[(eng.cohort == cohort) & (eng.variant == variant)], metric)
            beta, p = beta_perm(d, metric)
            ages = d.groupby('subject_id').age_at_session_days.mean()
            rows.append(dict(cohort=cohort, variant=variant, metric=metric, beta_per_100d=beta, p_two_sided=p,
                             mice=len(ages), age_min=ages.min(), age_max=ages.max(), neurons=len(d)))
within = pd.DataFrame(rows)
within.to_csv(OUT / '006_posthoc_within_cohort.csv', index=False)
print(within.round(4).to_string())

# (b) per ROI, all mice
rows = []
eng = both[(both.trial_set == 'eng') & (both.variant == 'paper')]
for metric in ['quench', 'ff_post']:
    for roi, d in eng.groupby('roi'):
        d = usable(d, metric)
        n = d.groupby('cohort').subject_id.nunique()
        if n.get('lifespan', 0) < MIN_MICE or n.get('bwm', 0) < MIN_MICE:
            continue
        beta, lo, hi = beta_boot(d, metric)
        rows.append(dict(metric=metric, roi=roi, beta_per_100d=beta, lo=lo, hi=hi, mice_bwm=n['bwm'],
                         mice_lifespan=n['lifespan'], neurons=len(d)))
roi_res = pd.DataFrame(rows)
roi_res.to_csv(OUT / '006_posthoc_roi.csv', index=False)
print(roi_res.round(4).to_string())

# figure
fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
mouse = conf[(conf.trial_set == 'eng') & (conf.variant == 'paper')].groupby(['subject_id', 'cohort']).agg(
    age=('age_at_session_days', 'mean'), quench=('quench', 'median'), ff_post=('ff_post', 'median')).reset_index()
for col, metric in enumerate(['quench', 'ff_post']):
    for cohort, color in [('bwm', '#2a78d6'), ('lifespan', '#d6552a')]:
        m = mouse[mouse.cohort == cohort]
        ax[col].scatter(m.age, m[metric], s=16, color=color, alpha=0.7, label=cohort)
        r = within[(within.cohort == cohort) & (within.variant == 'paper') & (within.metric == metric)].iloc[0]
        ax[col].text(0.02 if cohort == 'bwm' else 0.6, 0.95, f'{cohort}: β {r.beta_per_100d:.3f}/100 d, '
                     f'p {r.p_two_sided:.3f}', transform=ax[col].transAxes, color=color, fontsize=9, va='top')
    ax[col].set(xlabel='mouse mean age (d)', ylabel=f'mouse median {metric} (engaged)',
                title=f'POST HOC within-cohort age: {metric} (confirmation)')
    ax[col].legend(loc='lower right')
for col, metric in [(2, 'quench'), (3, 'ff_post')]:
    r = roi_res[roi_res.metric == metric].sort_values('beta_per_100d').reset_index(drop=True)
    ax[col].errorbar(r.beta_per_100d, np.arange(len(r)), xerr=[r.beta_per_100d - r.lo, r.hi - r.beta_per_100d],
                     fmt='o', color='k')
    ax[col].axvline(0, color='grey')
    ax[col].set(yticks=range(len(r)), yticklabels=r.roi + ' (' + r.mice_bwm.astype(str) + '/' +
                r.mice_lifespan.astype(str) + ')', xlabel='β_age per 100 d' + (' (log)' if metric in LOG_METRICS else ''),
                title=f'POST HOC per ROI: {metric} (all mice; bwm/lifespan mice)')
fig.tight_layout()
fig.savefig(OUT / '006_posthoc_checks.png', dpi=110)
