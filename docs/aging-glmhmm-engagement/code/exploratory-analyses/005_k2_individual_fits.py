"""Step 6b: same as 004 but with K = 2 and per-mouse state labels (user choice 2026-10-12).

With K = 3 the model alternated between solutions in which a biased mode was or was not stimulus-driven, so engaged
occupancy was unreliable (004_metric_risk_note.md). Here each mouse's engaged state is the state with the larger
stimulus weight in its own fit. Compared against the K = 3 metrics from 004 instead of refitting K = 2 vs K = 3.

Sections (each saves to artifacts/ and is skipped if its output exists; run each in its own process because JAX memory
grows across many fits):
  main      global K = 2 fit (best of 3 restarts) + per-mouse fits -> per-trial posteriors, per-mouse metrics
  seeds     two more global fits from single different seeds -> per-mouse metrics (sensitivity to local optima)
  sessions  mice with >= 2 sessions: per-mouse fits on alternate sessions (even vs odd) -> split-half reliability
  halves    every mouse: per-mouse fits on first vs second half of each session's trials -> reliability and drift
  plot      metric-definition and reliability figure
Age is deliberately not plotted here: metrics are judged on validity and reliability before looking at age (step 8).
"""
import json
import sys
import numpy as np
import pandas as pd
from glmhmm_metrics import (OUT, FEATURES, K, SEED, load_design, fit_global, fit_all_mice, engaged_states,
                            mouse_metrics)

A = OUT / 'artifacts'
dm, starts = load_design('exploration')
X, y = dm[FEATURES].values.astype(float), dm.y.values.astype(float)


def fit_and_measure(rows, n_states=K, seeds=None, glob=None):
    """Global fit on the selected trials (unless given), per-mouse fits, metrics. rows: boolean mask over dm.
    Session starts are recomputed on the subset so every kept run of trials starts a new HMM sequence."""
    sub = dm[rows].reset_index(drop=True)
    st = ((sub.session_id != sub.session_id.shift()) | (sub.trial_id.diff() != 1)).values
    Xs, ys = sub[FEATURES].values.astype(float), sub.y.values.astype(float)
    glob = fit_global(Xs, ys, st, n_states, seeds) if glob is None else glob
    eng = engaged_states(glob)
    models, post = fit_all_mice(glob, sub, st)
    return glob, eng, models, post, mouse_metrics(sub, post, models, eng, per_mouse_label=True), sub


section = sys.argv[1]
if section == 'main' and not (A / '005_mouse_metrics.csv').exists():
    glob, eng, models, post, metrics, _ = fit_and_measure(np.ones(len(dm), bool))
    out = dm[['subject_id', 'session_id', 'trial_id']].copy()
    for k in range(K):
        out[f'p_state{k}'] = post[:, k]
    # per-mouse label: engaged = the mouse's own highest-stimulus-weight state
    eng_of = dm.subject_id.map({m: int(np.argmax(np.asarray(md.coef_)[0])) for m, md in models.items()}).values
    out['p_engaged'] = post[np.arange(len(dm)), eng_of]
    out.to_parquet(A / '005_posteriors.parquet', index=False)
    metrics.to_csv(A / '005_mouse_metrics.csv', index=False)
    per_mouse = pd.DataFrame([dict(subject_id=m, state=k, stim_weight=float(np.asarray(md.coef_)[0, k]),
                                   intercept=float(np.asarray(md.intercept_)[k]))
                              for m, md in models.items() for k in range(K)])
    per_mouse.to_csv(A / '005_mouse_state_weights.csv', index=False)
    (A / '005_global_params.json').write_text(json.dumps(dict(
        coef=np.asarray(glob.coef_).tolist(), intercept=np.asarray(glob.intercept_).tolist(),
        transition=np.asarray(glob.transition_prob_).tolist(), engaged=eng.tolist())))
    print(metrics.round(3).to_string())

elif section == 'seeds' and not (A / '005_metrics_seeds.csv').exists():
    out = [fit_and_measure(np.ones(len(dm), bool), seeds=[SEED + s])[4].assign(seed=SEED + s) for s in (100, 200)]
    pd.concat(out).to_csv(A / '005_metrics_seeds.csv', index=False)

elif section == 'sessions' and not (A / '005_metrics_sessions.csv').exists():
    # Alternate sessions within mouse; mice with one session are dropped. The global fit is shared (all trials).
    session_rank = dm.groupby('subject_id').session_id.transform(lambda s: s.rank(method='dense').astype(int))
    multi = dm.groupby('subject_id').session_id.transform('nunique') >= 2
    glob = fit_global(X, y, starts)
    out = [fit_and_measure(multi & (session_rank % 2 == h), glob=glob)[4].assign(half=h) for h in (0, 1)]
    pd.concat(out).to_csv(A / '005_metrics_sessions.csv', index=False)

elif section == 'halves' and not (A / '005_metrics_halves.csv').exists():
    # First vs second half of every session's trials. Differences mix unreliability with within-session drift.
    position = dm.groupby('session_id').cumcount() / dm.groupby('session_id').y.transform('size')
    glob = fit_global(X, y, starts)
    out = [fit_and_measure((position >= 0.5) == bool(h), glob=glob)[4].assign(half=h) for h in (0, 1)]
    pd.concat(out).to_csv(A / '005_metrics_halves.csv', index=False)

elif section == 'plot':
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}
    met = pd.read_csv(A / '005_mouse_metrics.csv')
    post = pd.read_parquet(A / '005_posteriors.parquet')
    weights = pd.read_csv(A / '005_mouse_state_weights.csv')
    glob = json.loads((A / '005_global_params.json').read_text())
    d = dm.merge(post, on=['subject_id', 'session_id', 'trial_id'])
    fig, ax = plt.subplots(2, 5, figsize=(26, 9))

    # a, b: metric definition on one example session per cohort (the session closest to its cohort's median occupancy)
    for a, cohort in zip(ax[0, :2], COH):
        m = met[met.cohort == cohort]
        mouse = m.iloc[(m.occupancy - m.occupancy.median()).abs().argmin()].subject_id
        s = d[d.subject_id == mouse]
        s = s[s.session_id == s.session_id.iloc[0]]
        t = np.arange(len(s))
        a.plot(t, s.p_engaged, color=COH[cohort], lw=1.2, label='P(engaged)')
        a.fill_between(t, 0, 1, where=s.p_engaged.values <= 0.5, color='#8a8985', alpha=0.25, lw=0,
                       label='disengaged trial (P ≤ 0.5)')
        a.axhline(0.5, color='#0b0b0b', lw=0.8, ls=':')
        err = s.correct.values == 0
        a.plot(t[err], np.full(err.sum(), 1.04), '|', color='#0b0b0b', ms=5, label='error')
        r = m[m.subject_id == mouse].iloc[0]
        a.set(xlabel='trial', ylabel='P(engaged)', ylim=(0, 1.08),
              title=f'{cohort} example {mouse}: occupancy {r.occupancy:.2f}, mean disengaged bout '
                    f'{r.disengaged_bout:.1f}, {r.switches_per_100:.1f} switches/100')
        a.title.set_fontsize(9)
        a.legend(frameon=False, fontsize=7, loc='lower left', ncol=3)

    # c: per-mouse stimulus weight of the engaged vs disengaged state: are the two states well separated in every mouse?
    for cohort, c in COH.items():
        m = met[met.cohort == cohort]
        ax[0, 2].scatter(m.disengaged_stim_weight, m.engaged_stim_weight, color=c, s=24, label=cohort)
    lim = [min(0, met.disengaged_stim_weight.min()), met.engaged_stim_weight.max()]
    ax[0, 2].plot(lim, lim, color='#8a8985', ls=':', lw=1)
    ax[0, 2].set(xlabel='per-mouse disengaged-state stimulus weight', ylabel='per-mouse engaged-state stimulus weight',
                 title=f"State separation per mouse (global: {glob['coef'][0][0]:.1f} vs {glob['coef'][0][1]:.1f})")
    ax[0, 2].legend(frameon=False)

    # d: occupancy distributions, hard threshold vs soft
    bins = np.linspace(0, 1, 21)
    for cohort, c in COH.items():
        ax[0, 3].hist(met[met.cohort == cohort].occupancy, bins, color=c, alpha=0.6, label=f'{cohort} (n={sum(met.cohort == cohort)})')
    ax[0, 3].set(xlabel='engaged occupancy (P > 0.5)', ylabel='mice', title='Engaged occupancy per mouse')
    ax[0, 3].legend(frameon=False)
    ax[0, 4].scatter(met.occupancy, met.soft_occupancy, c=met.cohort.map(COH), s=24)
    ax[0, 4].plot([0, 1], [0, 1], color='#8a8985', ls=':', lw=1)
    ax[0, 4].set(xlabel='occupancy (P > 0.5)', ylabel='soft occupancy (mean P)', title='Threshold vs soft occupancy')

    # e-h: reliability; i: adversarial check on time in session
    def reliability(a, other, x_label, y_label, title, metric='occupancy'):
        """Scatter the main-fit metric against an alternative fit's metric, with Spearman rho."""
        j = other
        a.scatter(j[f'{metric}_x'], j[f'{metric}_y'], c=j.cohort.map(COH), s=24)
        lim = [min(j[f'{metric}_x'].min(), j[f'{metric}_y'].min()), max(j[f'{metric}_x'].max(), j[f'{metric}_y'].max())]
        a.plot(lim, lim, color='#8a8985', ls=':', lw=1)
        rho = spearmanr(j[f'{metric}_x'], j[f'{metric}_y']).statistic
        a.set(xlabel=x_label, ylabel=y_label, title=f'{title} (Spearman ρ = {rho:.2f}, n = {len(j)})')
        a.title.set_fontsize(9)

    seeds = pd.read_csv(A / '005_metrics_seeds.csv')
    j = met.merge(seeds[seeds.seed == seeds.seed.min()], on=['subject_id', 'cohort'])
    reliability(ax[1, 0], j, 'occupancy, main fit', 'occupancy, other global seed', 'Global-fit seed')
    sess = pd.read_csv(A / '005_metrics_sessions.csv')
    j = sess[sess.half == 0].merge(sess[sess.half == 1], on=['subject_id', 'cohort'])
    reliability(ax[1, 1], j, 'occupancy, even sessions', 'occupancy, odd sessions', 'Split by session')
    halves = pd.read_csv(A / '005_metrics_halves.csv')
    j = halves[halves.half == 0].merge(halves[halves.half == 1], on=['subject_id', 'cohort'])
    reliability(ax[1, 2], j, 'occupancy, first half of sessions', 'occupancy, second half', 'Split within session')
    k3 = pd.read_csv(A / '004_mouse_metrics.csv')
    j = met.merge(k3, on=['subject_id', 'cohort'])
    reliability(ax[1, 3], j, 'occupancy, K = 2 (this fit)', 'occupancy, K = 3 (step 6)', 'K = 2 vs K = 3')

    d['position'] = (d.trial_id // 100) * 100
    for cohort, c in COH.items():
        g = d[d.cohort == cohort].groupby(['position', 'subject_id']).p_engaged.mean().groupby('position')
        g = g.agg(['mean', 'sem', 'size'])
        g = g[g['size'] >= 3]
        ax[1, 4].errorbar(g.index, g['mean'], g['sem'], color=c, marker='o', ms=4, label=cohort)
    ax[1, 4].set(xlabel='trial number in session (100-trial bins)', ylabel='mean P(engaged)', ylim=(0, 1),
                 title='Engagement vs time in session')
    ax[1, 4].legend(frameon=False)
    for a in ax.flat:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'exploratory-analyses' / '005_k2_individual_fits.png', dpi=110)

    # Reliability of every metric, printed for the risk note
    for name, other, key in [('seed', seeds[seeds.seed == seeds.seed.min()], None), ('sessions', sess, 'half'),
                             ('halves', halves, 'half'), ('K3', k3, None)]:
        j = (met.merge(other, on=['subject_id', 'cohort']) if key is None
             else other[other[key] == 0].merge(other[other[key] == 1], on=['subject_id', 'cohort']))
        print(name, {mt: round(spearmanr(j[f'{mt}_x'], j[f'{mt}_y'], nan_policy='omit').statistic, 2) if f'{mt}_x' in j else None
                     for mt in ['occupancy', 'soft_occupancy', 'disengaged_bout', 'switches_per_100',
                                'engaged_stim_weight', 'disengaged_stim_weight']})
