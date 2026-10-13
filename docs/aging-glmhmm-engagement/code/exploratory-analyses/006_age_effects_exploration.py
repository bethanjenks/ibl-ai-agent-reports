"""Step 8: age effects on K = 2 engagement metrics, exploration mice only (48: 42 bwm, 6 lifespan).

Mouse = independent unit; mouse age = mean age at its sessions. Metrics from 005 (K = 2, per-mouse labels).
  H1        soft occupancy (mean P(engaged)) vs age
  H-eng     engaged-state stimulus weight vs age (co-primary)
  H3        does occupancy and/or engaged stimulus weight account for the age effect on the lapse proxy?
  H4        does occupancy relate to RT variability (sd_log_rt, from aging_rt_variability), beyond age?
  H2        disengaged bout length and switches per 100 trials vs age (descriptive only)
Checks: BWM only (age within one cohort), lab-centred, churchlandlab only, session-level mixed model with lab,
first session per mouse only, and reward history (Mohammadi et al. 2025: past rewards predict disengagement).
All p-values here are exploratory; nothing is a confirmatory test.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
A = OUT / 'artifacts'
RT = OUT.parents[1] / 'aging-rt-variability' / 'code' / 'artifacts' / '002_session_metrics_primary.csv'
N_BOOT, N_PERM, SEED = 2000, 5000, 0
REWARD_BIN = 25
EARLY_REWARDS = 100            # occupancy before satiety can build up: trials before the 100th reward
COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}
rng = np.random.default_rng(SEED)

met = pd.read_csv(A / '005_mouse_metrics.csv')
dm = pd.read_parquet(A / '001_design_matrix.parquet')
d = dm[dm.split == 'exploration'].merge(pd.read_parquet(A / '005_posteriors.parquet'),
                                        on=['subject_id', 'session_id', 'trial_id'])
d = d.sort_values(['session_id', 'trial_id'])
d['rewards_so_far'] = d.groupby('session_id').correct.cumsum() - d.correct     # rewards before this trial
rt = pd.read_csv(RT).groupby('subject_id').sd_log_rt.mean()
met['sd_log_rt'] = met.subject_id.map(rt)


def spearman_ci(x, y):
    """Spearman rho, 95% bootstrap CI (resampling mice) and two-sided permutation p (shuffling y)."""
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    boot = [spearmanr(x[i], y[i]).statistic for i in idx]
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(N_PERM)])
    return rho, *np.nanpercentile(boot, [2.5, 97.5]), np.mean(np.abs(perm) >= abs(rho))


def lab_centred(df, col):
    """Value minus its lab mean, so only within-lab variation remains."""
    return df[col] - df.groupby('lab')[col].transform('mean')


rows = []
metrics = ['soft_occupancy', 'occupancy', 'engaged_stim_weight', 'lapse_proxy', 'disengaged_bout', 'switches_per_100']
for mt in metrics:
    for label, sub in [('all', met), ('bwm only', met[met.cohort == 'bwm']),
                       ('churchlandlab', met[met.lab == 'churchlandlab'])]:
        sub = sub.dropna(subset=[mt])
        rows.append(dict(metric=mt, subset=label, n=len(sub), **dict(zip(['rho', 'ci_lo', 'ci_hi', 'p_perm'],
                                                                         spearman_ci(sub.age, sub[mt])))))
    lc = met.dropna(subset=[mt])
    rows.append(dict(metric=mt, subset='lab-centred', n=len(lc), **dict(zip(['rho', 'ci_lo', 'ci_hi', 'p_perm'],
                     spearman_ci(lab_centred(lc, 'age'), lab_centred(lc, mt))))))

# Session-level mixed model for occupancy: sessions nested in mice, lab as fixed effect
sess = (d.groupby('session_id').agg(subject_id=('subject_id', 'first'), lab=('lab', 'first'),
                                     age=('age_at_session_days', 'first'), soft_occupancy=('p_engaged', 'mean'))
        .reset_index())
sess['occ_z'] = (sess.soft_occupancy - sess.soft_occupancy.mean()) / sess.soft_occupancy.std()
sess['age_z'] = (sess.age - sess.age.mean()) / sess.age.std()
mixed = smf.mixedlm('occ_z ~ age_z + C(lab)', sess, groups=sess.subject_id).fit()
rows.append(dict(metric='soft_occupancy', subset='session mixed model (beta_age, z units)', n=len(sess),
                 rho=mixed.params['age_z'], ci_lo=mixed.conf_int().loc['age_z', 0],
                 ci_hi=mixed.conf_int().loc['age_z', 1], p_perm=mixed.pvalues['age_z']))

# First session per mouse only (equal weight per mouse, no averaging advantage for multi-session mice)
first = sess.merge(d.groupby('session_id').date.first().reset_index(), on='session_id')
first = first.sort_values('date').groupby('subject_id').first().reset_index()
rows.append(dict(metric='soft_occupancy', subset='first session only', n=len(first),
                 **dict(zip(['rho', 'ci_lo', 'ci_hi', 'p_perm'], spearman_ci(first.age, first.soft_occupancy)))))

# Reward history: occupancy over trials before the EARLY_REWARDS-th reward (little time for satiety)
early = d[d.rewards_so_far < EARLY_REWARDS].groupby('subject_id').p_engaged.mean()
met['early_soft_occupancy'] = met.subject_id.map(early)
rows.append(dict(metric='soft_occupancy', subset=f'before reward {EARLY_REWARDS}', n=len(met),
                 **dict(zip(['rho', 'ci_lo', 'ci_hi', 'p_perm'], spearman_ci(met.age, met.early_soft_occupancy)))))

# H3: does the age effect on lapses survive adjustment for occupancy and engaged stimulus weight? (z-scored OLS)
z = met[['age', 'lapse_proxy', 'soft_occupancy', 'engaged_stim_weight', 'sd_log_rt']].apply(lambda c: (c - c.mean()) / c.std())
h3 = {name: smf.ols(f, z).fit() for name, f in [('lapse ~ age', 'lapse_proxy ~ age'),
                                                ('lapse ~ age + occupancy', 'lapse_proxy ~ age + soft_occupancy'),
                                                ('lapse ~ age + engaged weight', 'lapse_proxy ~ age + engaged_stim_weight'),
                                                ('lapse ~ age + both', 'lapse_proxy ~ age + soft_occupancy + engaged_stim_weight')]}
# H4: RT variability vs occupancy, with and without age
h4 = {name: smf.ols(f, z.dropna()).fit() for name, f in [('sd_log_rt ~ occupancy', 'sd_log_rt ~ soft_occupancy'),
                                                         ('sd_log_rt ~ occupancy + age', 'sd_log_rt ~ soft_occupancy + age')]}
for name, fit in {**h3, **h4}.items():
    for term in fit.params.index[1:]:
        rows.append(dict(metric=name, subset=f'OLS beta: {term}', n=int(fit.nobs), rho=fit.params[term],
                         ci_lo=fit.conf_int().loc[term, 0], ci_hi=fit.conf_int().loc[term, 1], p_perm=fit.pvalues[term]))
stats = pd.DataFrame(rows)
stats.to_csv(A / '006_age_stats_exploration.csv', index=False)
print(stats.round(3).to_string())

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 4, figsize=(22, 9.5))


def scatter_age(a, col, ylabel, title, logy=False):
    """Metric vs mouse age; churchlandlab mice outlined in black; text = Spearman rho (all, bwm only)."""
    for cohort, c in COH.items():
        m = met[met.cohort == cohort]
        a.scatter(m.age, m[col], color=c, s=40, edgecolor=np.where(m.lab == 'churchlandlab', '#0b0b0b', 'white'),
                  lw=1.2, label=cohort)
    s = stats[(stats.metric == col)].set_index('subset')
    a.set(xlabel='mouse age (days)', ylabel=ylabel, yscale='log' if logy else 'linear',
          title=f"{title}\nρ all {s.loc['all', 'rho']:.2f} [{s.loc['all', 'ci_lo']:.2f}, {s.loc['all', 'ci_hi']:.2f}], "
                f"bwm only {s.loc['bwm only', 'rho']:.2f}, lab-centred {s.loc['lab-centred', 'rho']:.2f}")
    a.title.set_fontsize(9)


scatter_age(ax[0, 0], 'soft_occupancy', 'soft occupancy (mean P(engaged))', 'H1: time engaged')
ax[0, 0].legend(frameon=False, title='black edge = churchlandlab', title_fontsize=8)
scatter_age(ax[0, 1], 'engaged_stim_weight', 'engaged-state stimulus weight', 'Co-primary: stimulus following when engaged')
scatter_age(ax[0, 2], 'lapse_proxy', 'error rate at 100% contrast', 'Lapse proxy')

# H3: lapses vs the two candidate explanations
for a, col, xlabel in [(ax[0, 3], 'soft_occupancy', 'soft occupancy')]:
    for cohort, c in COH.items():
        m = met[met.cohort == cohort]
        a.scatter(m[col], m.lapse_proxy, color=c, s=40, edgecolor='white')
    b = {k: v.params.get('age', np.nan) for k, v in h3.items()}
    a.set(xlabel=xlabel, ylabel='error rate at 100% contrast',
          title='H3: age β on lapses (z units)\n' + ', '.join(f'{k.split("~ ")[1]}: {v:.2f}' for k, v in b.items()))
    a.title.set_fontsize(9)

# H4: RT variability vs occupancy
for cohort, c in COH.items():
    m = met[met.cohort == cohort]
    ax[1, 0].scatter(m.soft_occupancy, m.sd_log_rt, color=c, s=40, edgecolor='white')
ax[1, 0].set(xlabel='soft occupancy', ylabel='RT variability (SD of log RT)',
             title=f"H4: occupancy β {h4['sd_log_rt ~ occupancy'].params['soft_occupancy']:.2f}, "
                   f"with age {h4['sd_log_rt ~ occupancy + age'].params['soft_occupancy']:.2f} (z units)")
ax[1, 0].title.set_fontsize(9)

# Reward history: P(engaged) as rewards accumulate within a session
d['reward_bin'] = (d.rewards_so_far // REWARD_BIN) * REWARD_BIN
for cohort, c in COH.items():
    g = d[d.cohort == cohort].groupby(['reward_bin', 'subject_id']).p_engaged.mean().groupby('reward_bin')
    g = g.agg(['mean', 'sem', 'size'])
    g = g[g['size'] >= 3]
    ax[1, 1].errorbar(g.index, g['mean'], g['sem'], color=c, marker='o', ms=4, label=cohort)
ax[1, 1].axvline(EARLY_REWARDS, color='#8a8985', ls=':', lw=1)
ax[1, 1].set(xlabel=f'rewards earned so far in session ({REWARD_BIN}-reward bins)', ylabel='mean P(engaged)',
             ylim=(0, 1), title='Reward history (satiety check)')
ax[1, 1].legend(frameon=False)

# Robustness: first session only, and before the 100th reward
for a, x, y, title in [(ax[1, 2], first.age, first.soft_occupancy, 'first session per mouse only'),
                       (ax[1, 3], met.age, met.early_soft_occupancy, f'trials before reward {EARLY_REWARDS} only')]:
    cohort = first.subject_id.map(met.set_index('subject_id').cohort) if a is ax[1, 2] else met.cohort
    a.scatter(x, y, c=cohort.map(COH), s=40, edgecolor='white')
    s = stats[(stats.metric == 'soft_occupancy') & (stats.subset == ('first session only' if a is ax[1, 2]
                                                                    else f'before reward {EARLY_REWARDS}'))].iloc[0]
    a.set(xlabel='mouse age (days)', ylabel='soft occupancy',
          title=f'H1 check, {title}: ρ {s.rho:.2f} [{s.ci_lo:.2f}, {s.ci_hi:.2f}]')
    a.title.set_fontsize(9)
for a in ax.flat:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '006_age_effects_exploration.png', dpi=110)

# H2 descriptive, printed only
print(stats[stats.metric.isin(['disengaged_bout', 'switches_per_100'])].round(3).to_string())
