"""POST HOC SENSITIVITY (user request 2026-10-12), not confirmatory: rerun C1a and C3 on confirmation mice with
trial filters that remove anticipatory wheel movements.

  none   the locked analysis (all trials except no-go); C1a/C3 values recomputed from 008 outputs
  premove_out  drop trials with firstMovement < stimOn (and trials without firstMovement)
  zang_rt      keep trials with response time (response - stimOn) in 0.08-2 s (Zang et al. filter)
For each filter the K = 2 GLM-HMM is refit (global + per-mouse, frozen glmhmm_metrics.py). Inputs keep their original
trial history; every gap left by removed trials starts a new HMM segment. C3 uses the filtered easy trials among
the first 90 trials. Usage: python 012_sensitivity_trial_filters.py {premove_out|zang_rt|summary}
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / 'exploratory-analyses'))
from glmhmm_metrics import FEATURES, load_design, fit_global, fit_all_mice, engaged_states, mouse_metrics

A = OUT / 'artifacts'
DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
EARLY_WINDOW = 90

dm, _ = load_design('confirmation')
times = pd.read_parquet(DATA / 'trials.parquet', columns=['session_id', 'trial_id', 'stimOn_times',
                                                          'firstMovement_times', 'response_times'])
dm = dm.merge(times, on=['session_id', 'trial_id'], how='left')
KEEP = {'none': np.ones(len(dm), bool),
        'premove_out': (dm.firstMovement_times >= dm.stimOn_times).values,
        'zang_rt': (dm.response_times - dm.stimOn_times).between(0.08, 2).values}


def c3_metric(d):
    """Per-mouse easy-trial (|contrast| >= 0.25) accuracy in the first EARLY_WINDOW trials of each session."""
    e = d[(d.trial_id < EARLY_WINDOW) & (d.stim.abs() >= 0.25)]
    return e.groupby('subject_id').correct.mean()


name = sys.argv[1]
if name in ('premove_out', 'zang_rt') and not (A / f'012_metrics_{name}.csv').exists():
    sub = dm[KEEP[name]].reset_index(drop=True)
    starts = ((sub.session_id != sub.session_id.shift()) | (sub.trial_id.diff() != 1)).values
    X, y = sub[FEATURES].values.astype(float), sub.y.values.astype(float)
    glob = fit_global(X, y, starts)
    models, post = fit_all_mice(glob, sub, starts)
    met = mouse_metrics(sub, post, models, engaged_states(glob), per_mouse_label=True)
    met['acc_easy_early'] = met.subject_id.map(c3_metric(sub))
    met['fraction_trials_kept'] = met.subject_id.map(sub.groupby('subject_id').size() / dm.groupby('subject_id').size())
    met.to_csv(A / f'012_metrics_{name}.csv', index=False)
    print(name, 'global stim weights', np.asarray(glob.coef_)[0].round(2))

elif name == 'summary':
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    base = pd.read_csv(A / '008_mouse_metrics_confirmation.csv')
    base['acc_easy_early'] = base.subject_id.map(c3_metric(dm))
    base['fraction_trials_kept'] = 1.0
    runs = {'none (locked)': base, 'firstMovement ≥ stimOn': pd.read_csv(A / '012_metrics_premove_out.csv'),
            'RT 0.08–2 s (Zang)': pd.read_csv(A / '012_metrics_zang_rt.csv')}
    rows = []
    for label, m in runs.items():
        for metric in ['soft_occupancy', 'acc_easy_early', 'engaged_stim_weight']:
            for scope, mm in [('all labs', m), ('churchlandlab only', m[m.lab == 'churchlandlab'])]:
                life, bwm = mm[mm.cohort == 'lifespan'][metric].dropna(), mm[mm.cohort == 'bwm'][metric].dropna()
                u = mannwhitneyu(life, bwm, alternative='less')
                rows.append(dict(filter=label, metric=metric, scope=scope, n=f'{len(life)} vs {len(bwm)}',
                                 lifespan_median=life.median(), bwm_median=bwm.median(),
                                 difference=life.median() - bwm.median(), auc=u.statistic / (len(life) * len(bwm)),
                                 p_one_sided=u.pvalue))
        rows.append(dict(filter=label, metric='fraction of trials kept', scope='lifespan / bwm median',
                         lifespan_median=m[m.cohort == 'lifespan'].fraction_trials_kept.median(),
                         bwm_median=m[m.cohort == 'bwm'].fraction_trials_kept.median()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'confirmatory-analyses' / '012_sensitivity_results.csv', index=False)
    print(res.round(4).to_string())

    COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, metric, ylabel in [(ax[0], 'soft_occupancy', 'soft occupancy (C1a metric)'),
                              (ax[1], 'acc_easy_early', 'easy-trial accuracy, first 90 trials (C3 metric)')]:
        for i, (label, m) in enumerate(runs.items()):
            for j, (cohort, c) in enumerate(COH.items()):
                v = m[m.cohort == cohort][metric].dropna()
                x = i + (j - 0.5) * 0.3
                a.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)), v, color=c,
                          s=10, alpha=0.5, label=cohort if i == 0 else None)
                a.plot([x - 0.1, x + 0.1], [v.median()] * 2, color='#0b0b0b', lw=2)
        a.set_xticks(range(len(runs)), list(runs), fontsize=9)
        a.set(ylabel=ylabel, title=f'POST HOC sensitivity: {ylabel.split(" (")[0]}')
        a.title.set_fontsize(10)
        a.spines[['top', 'right']].set_visible(False)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'confirmatory-analyses' / '012_sensitivity_trial_filters.png', dpi=110)
