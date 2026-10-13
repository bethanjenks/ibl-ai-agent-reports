"""Run the locked confirmatory tests on confirmation mice (exploratory-analyses/004_locked_test_proposal.md).

P1: Spearman ρ(mouse age, mouse sd_log_rt) > 0, all confirmation mice, one-sided permutation test.
S1: P1 in BWM mice only.  S2: sd_log_rt_z ~ age_z + C(lab) + (1 | mouse), β_age > 0.
S3: P1 with CV.  S4: ρ(age, median RT) > 0, BWM mice only.  S1–S4 Holm-corrected.
Trial rules and metric code are frozen (rt_metrics.py, primary defaults).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / 'exploratory-analyses'))
from rt_metrics import session_metrics, mouse_metrics

DATA = resolve_dataset_dir('ibl_aging_zang2025')
REQUIRED = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times', 'stimOn_times', 'firstMovement_times']
N_PERM = 10_000
COHORT_COLOR = {'bwm': 'C0', 'lifespan': 'C3'}
rng = np.random.default_rng(0)

# Per-trial table for confirmation mice, built exactly as in 001
split = pd.read_csv(OUT / 'artifacts' / 'split.csv')
sessions = pd.read_parquet(DATA / 'metadata' / 'sessions.parquet').merge(split[['subject_id', 'split']], on='subject_id')
sessions = sessions[sessions.split == 'confirmation']
trials = pd.read_parquet(DATA / 'metadata' / 'trials.parquet').merge(
    sessions[['session_id', 'subject_id', 'source_group', 'lab', 'age_at_session_days']], on='session_id'
).rename(columns={'source_group': 'cohort'})
trials['rt'] = trials.response_times - trials.stimOn_times
trials['contrast'] = trials.contrastLeft.fillna(0) + trials.contrastRight.fillna(0)
trials['complete'] = trials[REQUIRED].notna().all(axis=1) & (trials.choice != 0)
trials['early_move'] = trials.firstMovement_times < trials.stimOn_times
trials.to_parquet(OUT / 'artifacts' / '005_trials_rt_confirmation.parquet')

sess = session_metrics(trials)
mice = mouse_metrics(sess)
sess.to_csv(OUT / 'artifacts' / '005_session_metrics_confirmation.csv', index=False)
mice.to_csv(OUT / 'artifacts' / '005_mouse_metrics_confirmation.csv', index=False)
print(mice.groupby('cohort').agg(mice=('subject_id', 'size'), sessions=('n_sessions', 'sum')))


def perm_spearman(age, metric):
    """Observed Spearman ρ and one-sided permutation p for ρ > 0 (age shuffled across mice)."""
    rho = stats.spearmanr(age, metric)[0]
    null = np.array([stats.spearmanr(rng.permutation(age), metric)[0] for _ in range(N_PERM)])
    return rho, (1 + np.sum(null >= rho)) / (1 + N_PERM), null


def fmt_p(p):
    """p in scientific notation; '≤' when it is the permutation floor 1/(N_PERM+1)."""
    return f"≤ {p:.0e}" if np.isclose(p, 1 / (N_PERM + 1)) else f"= {p:.1e}" if p < 0.01 else f"= {p:.2f}"


bwm = mice[mice.cohort == 'bwm']
results, nulls = {}, {}
for name, d, metric in [('P1', mice, 'sd_log_rt'), ('S1', bwm, 'sd_log_rt'), ('S3', mice, 'cv'), ('S4', bwm, 'median_rt')]:
    rho, p, nulls[name] = perm_spearman(d.age_at_session_days.values, d[metric].values)
    results[name] = dict(test=name, mice=len(d), metric=metric, estimate=rho, p_one_sided=p)

z = lambda x: (x - x.mean()) / x.std()
sess['age_z'], sess['sd_log_rt_z'] = z(sess.age_at_session_days), z(sess.sd_log_rt)
fit = smf.mixedlm('sd_log_rt_z ~ age_z + C(lab)', sess, groups=sess.subject_id).fit(reml=True)
b, p2 = fit.params['age_z'], fit.pvalues['age_z']
results['S2'] = dict(test='S2', mice=sess.subject_id.nunique(), metric='sd_log_rt (β_age, mixed model + lab)',
                     estimate=b, p_one_sided=p2 / 2 if b > 0 else 1 - p2 / 2)
res = pd.DataFrame([results[k] for k in ['P1', 'S1', 'S2', 'S3', 'S4']])
sec = res.test != 'P1'
res.loc[sec, 'p_holm'] = multipletests(res.loc[sec, 'p_one_sided'], method='holm')[1]
res.loc[~sec, 'p_holm'] = res.loc[~sec, 'p_one_sided']
res['significant'] = res.p_holm < 0.05
res.to_csv(OUT / 'confirmatory-analyses' / '005_confirmatory_results.csv', index=False)
print(res.round(4).to_string())
print(fit.summary().tables[1].loc[['age_z']])

# Descriptive (no test): median RT across both cohorts
print('descriptive: ρ(age, median RT), all mice =', round(stats.spearmanr(mice.age_at_session_days, mice.median_rt)[0], 2))

fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
for a, (name, d, metric) in zip(ax[0], [('P1', mice, 'sd_log_rt'), ('S3', mice, 'cv'), ('S4', bwm, 'median_rt')]):
    for c, g in d.groupby('cohort'):
        a.scatter(g.age_at_session_days, g[metric], color=COHORT_COLOR[c], s=18, label=f'{c} (n={len(g)})')
    r = res.set_index('test').loc[name]
    a.set(xlabel='mouse mean age (days)', ylabel=metric,
          title=f"{name}: ρ = {r.estimate:+.2f}, p {fmt_p(r.p_one_sided)}")
    a.legend(fontsize=8)
a = ax[1, 0]
for c, g in bwm.groupby('cohort'):
    a.scatter(g.age_at_session_days, g.sd_log_rt, color=COHORT_COLOR[c], s=18)
r = res.set_index('test').loc['S1']
a.set(xlabel='mouse mean age (days)', ylabel='sd_log_rt', title=f"S1 (BWM only): ρ = {r.estimate:+.2f}, Holm p {fmt_p(r.p_holm)}")
a = ax[1, 1]
for name, color in [('P1', 'k'), ('S1', 'C0')]:
    a.hist(nulls[name], 60, histtype='step', color=color, label=f'{name} null')
    a.axvline(results[name]['estimate'], color=color, ls='--', label=f'{name} observed')
a.set(xlabel='Spearman ρ under shuffled age', ylabel='permutations', title='Permutation nulls')
a.legend(fontsize=8)
a = ax[1, 2]
a.errorbar(res.estimate, range(len(res)), fmt='o')
a.set_yticks(range(len(res)), [f"{t}: {m}" for t, m in zip(res.test, res.metric)], fontsize=8)
a.axvline(0, color='k', lw=0.6)
a.set(ylim=(len(res) - 0.5, -1), xlim=(-0.2, 0.7))  # inverted, with headroom for the labels
for i, r in res.iterrows():
    a.text(r.estimate, i - 0.25, f"Holm p {fmt_p(r.p_holm)}", fontsize=8, ha='center')
a.set(xlabel='estimate (ρ, or β for S2)', title='All locked tests (confirmation set)')
fig.tight_layout()
fig.savefig(OUT / 'confirmatory-analyses' / '005_confirmatory_tests.png', dpi=110)
