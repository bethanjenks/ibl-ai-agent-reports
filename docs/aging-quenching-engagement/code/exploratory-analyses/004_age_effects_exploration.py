"""Part B on exploration mice: do the Zang et al. age effects hold on engaged trials only?

Per neuron, five metrics (pre/post FR, pre/post FF, quench) on two equal-size random trial sets (N_REPEATS draws,
averaged), so FF trial-count bias is identical:
  eng  N engaged trials (P(engaged) > 0.8)
  all  N trials of any state (what the paper's analysis sees, at the same count)
Two window variants (chosen before looking at age):
  paper  post [0.16, 0.26] s, all trials, N = 150
  safe   post [0.02, 0.12] s, only trials with first movement > 0.12 s after stimOn, N = 100 (003: bwm only)
Pre window [-0.1, 0] s in both.

Age effect: β_age = slope per 100 days of mouse mean age, from OLS on neurons with ROI fixed effects
(log FR / log FF as in the paper; quench on natural scale). Mouse is the independent unit, so p comes from
permuting age across mice (neurons of a mouse stay together). Also run within churchlandlab (bwm vs lifespan).
Outputs: artifacts/004_partB_per_neuron.parquet, artifacts/004_age_betas.csv, 004_age_effects_exploration.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quench_common import PROJECT, PRE_END, load_trials, load_units, load_spikes, window_counts, state_metrics

VARIANTS = {'paper': dict(post_end=0.26, move_after=-np.inf, n=150),
            'safe': dict(post_end=0.12, move_after=0.12, n=100)}
METRICS = ['fr_pre', 'fr_post', 'ff_pre', 'ff_post', 'quench']
LOG_METRICS = ['fr_pre', 'fr_post', 'ff_pre', 'ff_post']
THETA, N_REPEATS, N_PERM = 0.8, 20, 2000
rng = np.random.default_rng(0)

units = load_units()
units = units[units.split == 'exploration']
trials = load_trials()
trials = trials[trials.session_id.isin(units.session_id)].sort_values(['session_id', 'trial_id'])
trials['mv'] = trials.firstMovement_times - trials.stimOn_times

rows = []
for session_id, st in trials.groupby('session_id'):
    su = pd.concat([u for _, u in units[units.session_id == session_id].groupby('recording_id')])
    for variant, v in VARIANTS.items():
        windows = np.array([PRE_END, v['post_end']])
        sv = st[st.mv > v['move_after']]
        pools = {'eng': np.flatnonzero(sv.p_engaged.values > THETA), 'all': np.arange(len(sv))}
        if len(pools['eng']) < v['n']:
            continue
        counts = np.concatenate([window_counts(*load_spikes(rec), u.unit_index.values, sv.stimOn_times.values, windows)
                                 for rec, u in su.groupby('recording_id')], axis=1)
        for trial_set, pool in pools.items():
            draws = [state_metrics(counts[d], sv.signed_contrast.values[d], windows, v['post_end'])
                     for d in (rng.choice(pool, v['n'], replace=False) for _ in range(N_REPEATS))]
            m = {k: np.mean([d[k] for d in draws], 0) for k in METRICS}
            rows.append(su[['session_id', 'recording_id', 'unit_index', 'roi', 'subject_id', 'lab', 'cohort',
                            'age_at_session_days']].assign(variant=variant, trial_set=trial_set, **m))
df = pd.concat(rows, ignore_index=True)
df.to_parquet(PROJECT / 'artifacts' / '004_partB_per_neuron.parquet')
print('sessions per variant/cohort:\n', df.groupby(['variant', 'cohort']).session_id.nunique())


def age_beta(d, metric, permute=False):
    """OLS slope of metric on mouse mean age (per 100 d) with ROI fixed effects; optionally ages permuted
    across mice. d: neuron rows. Returns slope."""
    mouse_age = d.groupby('subject_id').age_at_session_days.mean() / 100
    if permute:
        mouse_age = pd.Series(rng.permutation(mouse_age.values), mouse_age.index)
    y = np.log(d[metric].values) if metric in LOG_METRICS else d[metric].values
    X = np.column_stack([d.subject_id.map(mouse_age).values, pd.get_dummies(d.roi).values.astype(float)])
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


results = []
for (variant, trial_set), d in df.groupby(['variant', 'trial_set']):
    for scope, ds in [('all labs', d), ('churchlandlab', d[d.lab == 'churchlandlab'])]:
        for metric in METRICS:
            ds_m = ds[np.isfinite(ds[metric]) & ((ds[metric] > 0) | (metric not in LOG_METRICS))]
            beta = age_beta(ds_m, metric)
            null = np.array([age_beta(ds_m, metric, permute=True) for _ in range(N_PERM)])
            results.append(dict(variant=variant, trial_set=trial_set, scope=scope, metric=metric, beta=beta,
                                p_perm=(np.sum(np.abs(null) >= abs(beta)) + 1) / (N_PERM + 1),
                                null_sd=null.std(), mice=ds_m.subject_id.nunique(), neurons=len(ds_m)))
res = pd.DataFrame(results)
res.to_csv(PROJECT / 'artifacts' / '004_age_betas.csv', index=False)
print(res.pivot_table(index=['scope', 'variant', 'metric'], columns='trial_set', values=['beta', 'p_perm']).round(4))

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 5, figsize=(24, 9))
for col, metric in enumerate(METRICS):
    a = ax[0, col]
    r = res[res.metric == metric].copy()
    r['label'] = r.scope + '\n' + r.variant
    for i, (trial_set, color, dx) in enumerate([('all', '#888888', -0.15), ('eng', '#1baf7a', 0.15)]):
        rr = r[r.trial_set == trial_set].reset_index(drop=True)
        a.errorbar(np.arange(len(rr)) + dx, rr.beta, yerr=1.96 * rr.null_sd, fmt='o', color=color, label=trial_set)
        for x, (b, p) in enumerate(zip(rr.beta, rr.p_perm)):
            if p < 0.05:
                a.text(x + dx, b, '*', fontsize=14, ha='center', va='bottom')
    a.axhline(0, color='grey')
    a.set(xticks=range(len(rr)), title=f'{metric}: β_age per 100 d' + (' (log)' if metric in LOG_METRICS else ''),
          ylabel='β ± 1.96 × permutation SD')
    a.set_xticklabels(rr.scope + '\n' + rr.variant, fontsize=8)
    a.legend()
# bottom row: mouse-level means vs age (paper windows for FR/FF; quench in both variants)
mouse = df.groupby(['variant', 'trial_set', 'subject_id', 'cohort']).agg(
    age=('age_at_session_days', 'mean'), **{m: (m, 'median') for m in METRICS}).reset_index()
panels = [('paper', 'fr_pre'), ('paper', 'ff_post'), ('paper', 'quench'), ('safe', 'ff_post'), ('safe', 'quench')]
for col, (variant, metric) in enumerate(panels):
    a = ax[1, col]
    for trial_set, color in [('all', '#888888'), ('eng', '#1baf7a')]:
        d = mouse[(mouse.variant == variant) & (mouse.trial_set == trial_set)]
        for cohort, marker in [('bwm', 'o'), ('lifespan', '^')]:
            dc = d[d.cohort == cohort]
            a.scatter(dc.age, dc[metric], color=color, marker=marker, s=18, alpha=0.7,
                      label=f'{trial_set} {cohort}')
    a.set(xlabel='mouse mean age (d)', ylabel=f'mouse median {metric}', title=f'{variant} window: {metric}')
    if metric in LOG_METRICS:
        a.set_yscale('log')
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(PROJECT / 'exploratory-analyses' / '004_age_effects_exploration.png', dpi=110)
