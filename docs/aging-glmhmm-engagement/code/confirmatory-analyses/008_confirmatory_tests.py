"""Locked confirmatory tests on the 110 confirmation mice (exploratory-analyses/006_locked_test_proposal.md).

  fit    K = 2 global fit on confirmation mice (best of 3 restarts), per-mouse fits, per-mouse metrics -> artifacts
  test   precondition, primary fixed sequence C1a -> C2a -> C1b -> C2b (each one-sided alpha 0.05, stop at first
         failure), add-on model-free test C3, pre-declared descriptives, figure
Metric code is the frozen `exploratory-analyses/glmhmm_metrics.py` (checked by hash).
"""
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / 'exploratory-analyses'))
LOCKED_HASH = 'a11a42e7f6664355'
assert hashlib.sha256((OUT / 'exploratory-analyses' / 'glmhmm_metrics.py').read_bytes()).hexdigest()[:16] == LOCKED_HASH
from glmhmm_metrics import FEATURES, load_design, fit_global, fit_all_mice, engaged_states, mouse_metrics, ENGAGED_STIM_MIN

A = OUT / 'artifacts'
DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
ALPHA = 0.05
N_PERM, SEED = 10_000, 0
EARLY_WINDOW = 90                # C3: first 90 trials of each session (unbiased block)

dm, starts = load_design('confirmation')

if sys.argv[1] == 'fit' and not (A / '008_mouse_metrics_confirmation.csv').exists():
    X, y = dm[FEATURES].values.astype(float), dm.y.values.astype(float)
    glob = fit_global(X, y, starts)
    models, post = fit_all_mice(glob, dm, starts)
    met = mouse_metrics(dm, post, models, engaged_states(glob), per_mouse_label=True)
    met.to_csv(A / '008_mouse_metrics_confirmation.csv', index=False)
    eng_of = dm.subject_id.map({m: int(np.argmax(np.asarray(md.coef_)[0])) for m, md in models.items()}).values
    dm[['subject_id', 'session_id', 'trial_id']].assign(p_engaged=post[np.arange(len(dm)), eng_of]).to_parquet(
        A / '008_posteriors_confirmation.parquet', index=False)
    (A / '008_global_params_confirmation.json').write_text(json.dumps(dict(
        coef=np.asarray(glob.coef_).tolist(), intercept=np.asarray(glob.intercept_).tolist(),
        transition=np.asarray(glob.transition_prob_).tolist())))

elif sys.argv[1] == 'test':
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    from scipy.stats import mannwhitneyu, spearmanr
    rng = np.random.default_rng(SEED)
    met = pd.read_csv(A / '008_mouse_metrics_confirmation.csv')
    glob = json.loads((A / '008_global_params_confirmation.json').read_text())
    stim_w = np.array(glob['coef'][0])
    precondition = bool((stim_w > ENGAGED_STIM_MIN).sum() == 1 and (stim_w < ENGAGED_STIM_MIN).sum() == 1)
    print(f'precondition (one global state with stim weight > {ENGAGED_STIM_MIN}, one below): {precondition}; '
          f'stim weights {stim_w.round(2)}')
    assert precondition, 'precondition failed: stop and consult the user'

    # C3 metric: easy-trial accuracy in the first 90 trials (same definition as 007, window 90)
    early = dm[(dm.trial_id < EARLY_WINDOW) & (dm.stim.abs() >= 0.25)]
    met['acc_easy_early'] = met.subject_id.map(early.groupby('subject_id').correct.mean())
    life, bwm = met[met.cohort == 'lifespan'], met[met.cohort == 'bwm']

    def group_test(col):
        """One-sided Mann-Whitney U, lifespan < BWM. Returns statistic summary."""
        u = mannwhitneyu(life[col], bwm[col], alternative='less')
        return dict(lifespan_median=life[col].median(), bwm_median=bwm[col].median(),
                    effect=u.statistic / (len(life) * len(bwm)), p=u.pvalue, n=f'{len(life)} vs {len(bwm)}')

    def trend_test(col):
        """One-sided permutation test of Spearman rho(age, col) < 0 within the lifespan cohort."""
        rho = spearmanr(life.age, life[col]).statistic
        perm = np.array([spearmanr(rng.permutation(life.age.values), life[col]).statistic for _ in range(N_PERM)])
        return dict(effect=rho, p=(np.sum(perm <= rho) + 1) / (N_PERM + 1), n=len(life))

    sequence = [('C1a', 'soft occupancy: lifespan < BWM', lambda: group_test('soft_occupancy')),
                ('C2a', 'soft occupancy falls with age within lifespan', lambda: trend_test('soft_occupancy')),
                ('C1b', 'engaged stim weight: lifespan < BWM', lambda: group_test('engaged_stim_weight')),
                ('C2b', 'engaged stim weight falls with age within lifespan', lambda: trend_test('engaged_stim_weight'))]
    rows, gate_open = [], True
    for name, desc, fn in sequence:
        r = fn()
        significant = gate_open and r['p'] < ALPHA
        rows.append(dict(test=name, description=desc, evaluated=gate_open, significant=significant, **r))
        gate_open = significant                     # fixed sequence: later tests only if this one passed
    r = group_test('acc_easy_early')
    rows.append(dict(test='C3 (add-on)', description='easy-trial accuracy, first 90 trials: lifespan < BWM',
                     evaluated=True, significant=r['p'] < ALPHA, **r))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'confirmatory-analyses' / '008_confirmatory_results.csv', index=False)
    print(res.round(4).to_string())

    # ---------------- pre-declared descriptives (no tests) ----------------
    desc = []
    for col in ['soft_occupancy', 'engaged_stim_weight', 'lapse_proxy', 'acc_easy_early', 'disengaged_bout',
                'switches_per_100']:
        for label, sub in [('all', met), ('bwm only', bwm), ('lifespan only', life)]:
            s = sub.dropna(subset=[col])
            desc.append(dict(metric=col, subset=label, n=len(s), spearman_age=spearmanr(s.age, s[col]).statistic))
        cl = met[met.lab == 'churchlandlab']
        desc.append(dict(metric=col, subset='churchlandlab lifespan − bwm median', n=len(cl),
                         spearman_age=cl[cl.cohort == 'lifespan'][col].median() - cl[cl.cohort == 'bwm'][col].median()))
    z = met[['age', 'lapse_proxy', 'soft_occupancy', 'engaged_stim_weight']].apply(lambda c: (c - c.mean()) / c.std())
    for f in ['lapse_proxy ~ age', 'lapse_proxy ~ age + soft_occupancy', 'lapse_proxy ~ age + engaged_stim_weight',
              'lapse_proxy ~ age + soft_occupancy + engaged_stim_weight']:
        fit = smf.ols(f, z).fit()
        desc.append(dict(metric=f'H3 OLS: {f}', subset='beta_age (z)', n=int(fit.nobs), spearman_age=fit.params['age']))
    post = pd.read_parquet(A / '008_posteriors_confirmation.parquet').merge(dm, on=['subject_id', 'session_id', 'trial_id'])
    post = post.sort_values(['session_id', 'trial_id'])
    post['rewards_so_far'] = post.groupby('session_id').correct.cumsum() - post.correct
    early_occ = post[post.rewards_so_far < 100].groupby('subject_id').p_engaged.mean()
    met['early_soft_occupancy'] = met.subject_id.map(early_occ)
    desc.append(dict(metric='soft occupancy before reward 100', subset='lifespan − bwm median', n=len(met),
                     spearman_age=met[met.cohort == 'lifespan'].early_soft_occupancy.median()
                     - met[met.cohort == 'bwm'].early_soft_occupancy.median()))
    desc = pd.DataFrame(desc)
    desc.to_csv(OUT / 'confirmatory-analyses' / '008_descriptives.csv', index=False)
    print(desc.round(3).to_string())

    # ---------------- figure ----------------
    COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    for a, col, ylabel, tests in [(ax[0], 'soft_occupancy', 'soft occupancy (mean P(engaged))', ['C1a', 'C2a']),
                                  (ax[1], 'engaged_stim_weight', 'engaged-state stimulus weight', ['C1b', 'C2b']),
                                  (ax[2], 'acc_easy_early', 'easy-trial accuracy, first 90 trials', ['C3 (add-on)'])]:
        for cohort, c in COH.items():
            m = met[met.cohort == cohort]
            a.scatter(m.age, m[col], color=c, s=36, edgecolor=np.where(m.lab == 'churchlandlab', '#0b0b0b', 'white'),
                      lw=1.1, label=cohort)
        r = res.set_index('test')
        a.set(xlabel='mouse age (days)', ylabel=ylabel,
              title='; '.join(f"{t}: {'p = %.3g' % r.loc[t, 'p'] if r.loc[t, 'evaluated'] else 'not evaluated'}"
                              + (' ✓' if r.loc[t, 'significant'] else '') for t in tests))
        a.title.set_fontsize(9)
    ax[0].legend(frameon=False, title='black edge = churchlandlab', title_fontsize=8)
    post['reward_bin'] = (post.rewards_so_far // 25) * 25
    for cohort, c in COH.items():
        g = post[post.cohort == cohort].groupby(['reward_bin', 'subject_id']).p_engaged.mean().groupby('reward_bin')
        g = g.agg(['mean', 'sem', 'size'])
        g = g[g['size'] >= 3]
        ax[3].errorbar(g.index, g['mean'], g['sem'], color=c, marker='o', ms=4, label=cohort)
    ax[3].set(xlabel='rewards earned so far in session (25-reward bins)', ylabel='mean P(engaged)', ylim=(0, 1),
              title='Descriptive: reward history (satiety check)')
    ax[3].legend(frameon=False)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'confirmatory-analyses' / '008_confirmatory_tests.png', dpi=110)
