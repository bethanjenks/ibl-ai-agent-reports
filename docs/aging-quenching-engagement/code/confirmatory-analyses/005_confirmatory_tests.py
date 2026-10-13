"""Locked tests (004_locked_test_proposal.md, amended by user 2026-10-13 to test firing rate and Fano factor).

Usage: python 005_confirmatory_tests.py [confirmation|exploration]; 'exploration' is only a code check (should
reproduce 004) and writes files with an `_exploration_check` suffix.

Per neuron: metrics on equal-size random sets of engaged trials (P(engaged) > 0.8), via `partB_metrics`.
Statistic: β_age = OLS slope on mouse mean age (per 100 d) with ROI fixed effects (log FR / log FF, quench natural).
p: 10,000 permutations of age across mice.
Primary family Q (Holm, α 0.05), quench, one-sided β > 0 (less quenching with age):
  C1 paper windows, churchlandlab only;  C2 paper windows, all labs;  C3 movement-safe window, all labs
Secondary family F (Holm, α 0.05), paper windows, all labs:
  F1 fr_pre > 0;  F2 fr_post > 0;  F3 ff_post > 0;  F4 ff_pre two-sided (paper: no age effect)
Descriptive: β_eng, β_all, ratio β_eng / β_all with 95% mouse-bootstrap CIs, all metrics × windows × scopes.
Interpretation rule: "also when engaged" if any Q test is significant; "not explained by disengagement" if in
addition the upper 95% bound of C2's Δ = β_all − β_eng is below 0.5 × β_all (amendment, replaces ratio rule).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exploratory-analyses'))
from quench_common import PROJECT, METRICS, LOG_METRICS, load_trials, load_units, partB_metrics, age_slopes

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'confirmation'
SUFFIX = '' if SPLIT == 'confirmation' else '_exploration_check'
N_PERM, N_BOOT, CHUNK = 10_000, 2_000, 500
TESTS = [  # (family, name, metric, variant, scope, alternative)
    ('Q', 'C1', 'quench', 'paper', 'churchlandlab', 'greater'),
    ('Q', 'C2', 'quench', 'paper', 'all labs', 'greater'),
    ('Q', 'C3', 'quench', 'safe', 'all labs', 'greater'),
    ('F', 'F1', 'fr_pre', 'paper', 'all labs', 'greater'),
    ('F', 'F2', 'fr_post', 'paper', 'all labs', 'greater'),
    ('F', 'F3', 'ff_post', 'paper', 'all labs', 'greater'),
    ('F', 'F4', 'ff_pre', 'paper', 'all labs', 'two-sided')]
rng = np.random.default_rng(2026)
OUT = PROJECT / 'confirmatory-analyses'

artifact = PROJECT / 'artifacts' / f'005_partB_per_neuron_{SPLIT}.parquet'
if artifact.exists():
    df = pd.read_parquet(artifact)
else:
    units = load_units()
    df = partB_metrics(units[units.split == SPLIT], load_trials())
    df.to_parquet(artifact)
print(df[df.trial_set == 'eng'].groupby(['variant', 'cohort']).agg(
    sessions=('session_id', 'nunique'), mice=('subject_id', 'nunique'), neurons=('unit_index', 'size')))


def subset(metric, variant, scope, trial_set):
    """Neuron rows usable for one metric (finite; positive if log-scaled)."""
    d = df[(df.variant == variant) & (df.trial_set == trial_set)]
    if scope == 'churchlandlab':
        d = d[d.lab == 'churchlandlab']
    return d[np.isfinite(d[metric]) & ((d[metric] > 0) | (metric not in LOG_METRICS))]


def mouse_ages(d):
    """Mouse mean age (100 d), ordered as np.unique(d.subject_id)."""
    return d.groupby('subject_id').age_at_session_days.mean().sort_index().values / 100


def permutation_null(d, metric):
    ages = mouse_ages(d)
    return np.concatenate([age_slopes(d, metric, np.stack([rng.permutation(ages) for _ in range(CHUNK)]))
                           for _ in range(N_PERM // CHUNK)])


def weighted_slopes(d, metric, weights):
    """β_age with ROI fixed effects, neurons weighted by their mouse's bootstrap count.

    Weighting a mouse by k equals duplicating its neurons k times. weights: (n_sets, n_mice), mice ordered as
    np.unique(d.subject_id). Returns (n_sets,) slopes.
    """
    y = np.log(d[metric].values) if metric in LOG_METRICS else d[metric].values
    roi = pd.factorize(d.roi)[0]
    mouse = np.unique(d.subject_id, return_inverse=True)[1]
    a = mouse_ages(d)[mouse]
    R = np.eye(roi.max() + 1)[roi]                                # (n_neurons, n_roi) one-hot
    W = weights[:, mouse]                                         # (n_sets, n_neurons)
    WR = W @ R
    # an ROI whose mice were all left out of a draw has weight 0; its (unused) mean is set to 0
    roi_mean = lambda x: np.divide((W * x) @ R, WR, out=np.zeros_like(WR), where=WR > 0)
    y_res = y - roi_mean(y)[:, roi]
    a_res = a - roi_mean(a)[:, roi]
    return (W * a_res * y_res).sum(1) / (W * a_res ** 2).sum(1)


def bootstrap(d_eng, d_all, metric):
    """Mouse bootstrap of β_eng, β_all, their ratio and Δ = β_all − β_eng (same mice for both). (N_BOOT, 4).
    d_eng and d_all contain the same neurons, so they share mouse order."""
    n_mice = d_eng.subject_id.nunique()
    weights = rng.multinomial(n_mice, np.full(n_mice, 1 / n_mice), size=N_BOOT).astype(float)
    b_eng, b_all = weighted_slopes(d_eng, metric, weights), weighted_slopes(d_all, metric, weights)
    return np.column_stack([b_eng, b_all, b_eng / b_all, b_all - b_eng])


def holm(p):
    """Holm-adjusted p-values."""
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * np.asarray(p)[order]).clip(max=1)
    return adj[np.argsort(order)]


# ---------------- locked tests ----------------
rows, nulls = [], {}
for family, name, metric, variant, scope, alternative in TESTS:
    d = subset(metric, variant, scope, 'eng')
    beta = age_slopes(d, metric, mouse_ages(d)[None])[0]
    null = permutation_null(d, metric)
    nulls[name] = null
    p = ((null >= beta).sum() + 1) / (N_PERM + 1) if alternative == 'greater' else \
        ((np.abs(null) >= abs(beta)).sum() + 1) / (N_PERM + 1)
    rows.append(dict(family=family, test=name, metric=metric, variant=variant, scope=scope, alternative=alternative,
                     beta_per_100d=beta, z=beta / null.std(), p=p, mice=d.subject_id.nunique(), neurons=len(d)))
res = pd.DataFrame(rows)
res['p_holm'] = res.groupby('family').p.transform(lambda p: holm(p.values))
res['significant'] = res.p_holm < 0.05
res.to_csv(OUT / f'005_confirmatory_results{SUFFIX}.csv', index=False)
print(res.round(4).to_string())

# ---------------- descriptive: engaged vs all ----------------
desc = []
for metric in METRICS:
    for variant in ['paper', 'safe']:
        for scope in ['all labs', 'churchlandlab']:
            d_eng, d_all = subset(metric, variant, scope, 'eng'), subset(metric, variant, scope, 'all')
            keys = ['session_id', 'recording_id', 'unit_index']
            common = d_eng[keys].merge(d_all[keys])  # same neurons in both sets
            d_eng, d_all = d_eng.merge(common), d_all.merge(common)
            b_eng = age_slopes(d_eng, metric, mouse_ages(d_eng)[None])[0]
            b_all = age_slopes(d_all, metric, mouse_ages(d_all)[None])[0]
            boot = bootstrap(d_eng, d_all, metric)
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            desc.append(dict(metric=metric, variant=variant, scope=scope, beta_eng=b_eng, beta_eng_lo=lo[0],
                             beta_eng_hi=hi[0], beta_all=b_all, beta_all_lo=lo[1], beta_all_hi=hi[1],
                             ratio=b_eng / b_all, ratio_lo=lo[2], ratio_hi=hi[2],
                             delta=b_all - b_eng, delta_lo=lo[3], delta_hi=hi[3]))
desc = pd.DataFrame(desc)
desc.to_csv(OUT / f'005_engaged_vs_all{SUFFIX}.csv', index=False)
print(desc.round(3).to_string())
c2 = desc[(desc.metric == 'quench') & (desc.variant == 'paper') & (desc.scope == 'all labs')].iloc[0]
print(f"\nAlso when engaged (any Q significant after Holm): {res[res.family == 'Q'].significant.any()}")
print(f"C2 Δ = β_all − β_eng = {c2.delta:.4f} [{c2.delta_lo:.4f}, {c2.delta_hi:.4f}], β_all = {c2.beta_all:.4f}; "
      f"not explained by disengagement (Δ upper < 0.5 β_all): {c2.delta_hi < 0.5 * c2.beta_all}")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw=dict(width_ratios=[1.2, 1.3, 1]))
for i, r in res.iterrows():
    ax[0].violinplot(nulls[r.test] / nulls[r.test].std(), [i], showextrema=False)
    ax[0].plot(i, r.z, 'o', color='#d6552a' if r.significant else 'k', ms=8)
    ax[0].text(i, r.z + 0.3, f'p_holm {r.p_holm:.3g}', ha='center', fontsize=8)
ax[0].axhline(0, color='grey')
ax[0].set(xticks=range(len(res)), ylabel='β_age / permutation SD',
          title=f'Locked tests ({SPLIT}, engaged trials); red = significant after Holm')
ax[0].set_xticklabels(res.test + '\n' + res.metric + '\n' + res.variant + '\n' + res.scope, fontsize=8)
d = desc.reset_index(drop=True)
ax[1].errorbar(np.arange(len(d)), d.ratio, yerr=[d.ratio - d.ratio_lo, d.ratio_hi - d.ratio], fmt='o', color='k')
ax[1].axhline(1, color='#1baf7a', label='1: age effect unchanged on engaged trials')
ax[1].axhline(0.5, color='grey', ls='--', label='0.5 (interpretation threshold)')
ax[1].axhline(0, color='#9b59b6', label='0: fully explained by disengagement')
ax[1].set(xticks=range(len(d)), ylim=(-1, 3), ylabel='β_eng / β_all (95% mouse bootstrap)',
          title='Age effect on engaged trials relative to all trials')
ax[1].set_xticklabels(d.metric + '\n' + d.variant + '\n' + d.scope.str.replace('churchlandlab', 'CL'), fontsize=7)
ax[1].legend(fontsize=8)
mouse = df[(df.variant == 'paper')].groupby(['trial_set', 'subject_id', 'cohort']).agg(
    age=('age_at_session_days', 'mean'), quench=('quench', 'median')).reset_index()
for trial_set, color in [('all', '#888888'), ('eng', '#1baf7a')]:
    for cohort, marker in [('bwm', 'o'), ('lifespan', '^')]:
        m = mouse[(mouse.trial_set == trial_set) & (mouse.cohort == cohort)]
        ax[2].scatter(m.age, m.quench, color=color, marker=marker, s=18, alpha=0.7, label=f'{trial_set} {cohort}')
ax[2].axhline(0, color='grey', lw=0.5)
ax[2].set(xlabel='mouse mean age (d)', ylabel='mouse median quench (paper windows)', title='Quench vs age')
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / f'005_confirmatory_tests{SUFFIX}.png', dpi=110)
