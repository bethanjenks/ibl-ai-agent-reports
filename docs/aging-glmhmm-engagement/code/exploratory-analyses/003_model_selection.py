"""Model selection for the GLM-HMM by cross-validation on exploration mice.

What is scored is the full procedure we will use: a global fit on all exploration mice, then per-mouse fits started
from the global parameters. Each session is cut into N_FOLDS contiguous trial chunks; fold f holds out chunk f of
every session. Held-out chunks (rather than held-out sessions) let mice with a single session be scored. The HMM is
reset at every chunk boundary, in training and in testing, so no state sequence crosses a gap.

Stage 1: K = 3, Ridge strength x transition prior. Stage 2: K = 1..4 and raw vs tanh contrast at the best stage-1
setting (chosen by mean held-out log-likelihood per trial; pass it with --stage2 RIDGE PRIOR).
Each (setting, fold) is one work unit, saved as one JSON in artifacts/003_cv/ so the run can resume and be extended.
The global fit is restarted N_RESTARTS times; the restart with the best training log-likelihood is kept.

Usage:  python 003_model_selection.py --pilot              # one unit, serial, for timing
        python 003_model_selection.py --stage1             # 45 units, parallel
        python 003_model_selection.py --stage2 0.5 sticky  # K sweep and stimulus transform
        python 003_model_selection.py --plot               # summary figure
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
CV_DIR = OUT / 'artifacts' / '003_cv'
N_FOLDS = 5
N_RESTARTS = 3
N_WORKERS = 2                                 # 4 workers ran out of memory on a 15 GB machine
RIDGES = [0.1, 0.5, 2.0]                      # inverse prior variance on GLM weights
PRIORS = {'flat': (1.0, 1.0),                 # Dirichlet alpha (diagonal, off-diagonal) on transition rows
          'alpha2': (2.0, 2.0),               # Ashwood-like
          'sticky': (10.0, 1.0)}              # favours staying in a state
SEED = 20261012


def load_design(stim_column):
    """Exploration trials sorted by session and trial, with chunk index per trial.
    Returns X (n, 4), y (n,), session ids (n,), mouse ids (n,), chunk index (n,), cohort per mouse (Series)."""
    dm = pd.read_parquet(OUT / 'artifacts' / '001_design_matrix.parquet')
    dm = dm[dm.split == 'exploration'].sort_values(['subject_id', 'date', 'session_id', 'trial_id'])
    position = dm.groupby('session_id').cumcount()
    n_in_session = dm.groupby('session_id').y.transform('size')
    chunk = (position * N_FOLDS // n_in_session).values               # 0..N_FOLDS-1, contiguous within session
    X = dm[[stim_column, 'prev_choice', 'wsls', 'block']].values.astype(float)
    return X, dm.y.values.astype(float), dm.session_id.values, dm.subject_id.values, chunk, \
        dm.groupby('subject_id').cohort.first()


def segment_starts(session, chunk):
    """True at the first trial of every contiguous (session, chunk) run of the selected trials."""
    return (session != np.roll(session, 1)) | (chunk != np.roll(chunk, 1))


def run_unit(unit):
    """Fit one (setting, fold): global fit on training chunks of all mice (best of N_RESTARTS), per-mouse fits started
    from it, and per-mouse held-out log-likelihood. Writes and returns a dict."""
    os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    import jax
    import nemos as nmo
    K, ridge, prior, stim, fold = unit['K'], unit['ridge'], unit['prior'], unit['stim'], unit['fold']
    X, y, session, mouse, chunk, _ = load_design(stim)
    a_diag, a_off = PRIORS[prior]
    alpha = np.full((K, K), a_off) + np.eye(K) * (a_diag - a_off)

    def model(seed):
        return nmo.glm_hmm.GLMHMM(n_states=K, regularizer_strength=ridge, dirichlet_transition_proba=alpha,
                                  seed=jax.random.PRNGKey(seed))

    t0 = time.time()
    train, test = chunk != fold, chunk == fold
    starts_train = segment_starts(session[train], chunk[train])
    fits = [model(SEED + r).fit(X[train], y[train], session_starts=starts_train) for r in range(N_RESTARTS)]
    train_ll = [float(f.score(X[train], y[train], session_starts=starts_train)) for f in fits]
    glob = fits[int(np.argmax(train_ll))]
    init = (glob.coef_, glob.intercept_, glob.scale_, glob.initial_prob_, glob.transition_prob_)
    rows = []
    for m in np.unique(mouse):
        tr, te = train & (mouse == m), test & (mouse == m)
        fit_m = model(SEED).fit(X[tr], y[tr], init_params=init, session_starts=segment_starts(session[tr], chunk[tr]))
        rows.append(dict(subject_id=m, n_test=int(te.sum()),
                         test_ll=float(fit_m.score(X[te], y[te], session_starts=segment_starts(session[te], chunk[te]))),
                         test_ll_global=float(glob.score(X[te], y[te], session_starts=segment_starts(session[te], chunk[te])))))
    result = dict(unit, seconds=time.time() - t0, train_ll_restarts=train_ll,
                  global_coef=np.asarray(glob.coef_).tolist(), global_intercept=np.asarray(glob.intercept_).tolist(),
                  global_transition=np.asarray(glob.transition_prob_).tolist(), mice=rows)
    (CV_DIR / f"{unit['name']}.json").write_text(json.dumps(result))
    return result


def make_units(Ks, ridges, priors, stims):
    """All (setting, fold) units not already saved."""
    units = [dict(K=K, ridge=r, prior=p, stim=s, fold=f, name=f'K{K}_ridge{r}_{p}_{s}_fold{f}')
             for K in Ks for r in ridges for p in priors for s in stims for f in range(N_FOLDS)]
    return [u for u in units if not (CV_DIR / f"{u['name']}.json").exists()]


def run_parallel(units):
    print(f'{len(units)} units to run on {N_WORKERS} workers', flush=True)
    # one unit per process: JAX memory grows across units in a long-lived worker and the run was killed twice
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context('spawn'), max_tasks_per_child=1) as pool:
        for r in pool.map(run_unit, units):
            print(f"{r['name']}: {r['seconds']:.0f} s", flush=True)


def summarize():
    """Held-out log-likelihood per trial for every setting, overall and per cohort (mouse-level means too)."""
    _, _, _, _, _, cohort = load_design('stim')
    rows = []
    for f in CV_DIR.glob('*.json'):
        r = json.loads(f.read_text())
        for m in r['mice']:
            rows.append(dict(K=r['K'], ridge=r['ridge'], prior=r['prior'], stim=r['stim'], fold=r['fold'], **m))
    d = pd.DataFrame(rows)
    d['cohort'] = d.subject_id.map(cohort)
    keys = ['K', 'ridge', 'prior', 'stim']
    per_mouse = d.groupby(keys + ['cohort', 'subject_id'])[['test_ll', 'test_ll_global', 'n_test']].sum()
    per_mouse['ll_per_trial'] = per_mouse.test_ll / per_mouse.n_test
    per_mouse['ll_per_trial_global'] = per_mouse.test_ll_global / per_mouse.n_test
    per_mouse = per_mouse.reset_index()
    per_mouse.to_csv(OUT / 'artifacts' / '003_cv_per_mouse.csv', index=False)
    return per_mouse


if __name__ == '__main__':
    CV_DIR.mkdir(exist_ok=True)
    arg = sys.argv[1]
    if arg == '--pilot':
        r = run_unit(make_units([3], [0.5], ['alpha2'], ['stim'])[0] if make_units([3], [0.5], ['alpha2'], ['stim'])
                     else dict(K=3, ridge=0.5, prior='alpha2', stim='stim', fold=0, name='K3_ridge0.5_alpha2_stim_fold0'))
        print(f"pilot unit: {r['seconds']:.0f} s single-threaded, train LL per restart {np.round(r['train_ll_restarts'], 1)}")
    elif arg == '--stage1':
        run_parallel(make_units([3], RIDGES, list(PRIORS), ['stim']))
    elif arg == '--stage2':
        ridge, prior = float(sys.argv[2]), sys.argv[3]
        run_parallel(make_units([1, 2, 3, 4], [ridge], [prior], ['stim']) + make_units([3], [ridge], [prior], ['stim_tanh']))
    elif arg == '--ridge-ext-and-stage2':
        # NeMoS 0.2.9 never passes the Dirichlet priors to its M-step (hmm/expectation_maximization.py:647), so all
        # transition priors are equivalent; use 'flat'. Ridge 0.1 was best at the grid edge, so extend downwards,
        # then run stage 2 at the Ridge with the best mean per-mouse held-out log-likelihood.
        run_parallel(make_units([3], [0.01, 0.03], ['flat'], ['stim']))
        pm = summarize()
        by_ridge = pm[(pm.K == 3) & (pm.prior == 'flat') & (pm.stim == 'stim')].groupby('ridge').ll_per_trial.mean()
        print(by_ridge.round(5), flush=True)
        best = float(by_ridge.idxmax())
        print(f'best ridge {best}', flush=True)
        run_parallel(make_units([1, 2, 4], [best], ['flat'], ['stim']) + make_units([3], [best], ['flat'], ['stim_tanh']))
    elif arg == '--plot':
        import matplotlib.pyplot as plt
        BEST_RIDGE = 0.1                                             # chosen by --ridge-ext-and-stage2
        pm = summarize()
        pm = pm[pm.prior == 'flat']                                  # priors are inert in NeMoS 0.2.9; 'flat' covers all
        pm['bits'] = pm.ll_per_trial / np.log(2)                     # held-out log-likelihood, bits per trial
        pm['bits_global'] = pm.ll_per_trial_global / np.log(2)
        print(pm.groupby(['K', 'ridge', 'stim', 'cohort'])[['bits', 'bits_global']].mean().unstack('cohort').round(4))
        COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}
        fig, ax = plt.subplots(1, 5, figsize=(25, 4.6))

        def paired_delta(d, ref_mask, col='bits'):
            """Per-mouse difference from the reference rows (paired within mouse); returns the column 'delta'."""
            ref = d[ref_mask].set_index('subject_id')[col]
            return d.assign(delta=d[col] - d.subject_id.map(ref))

        # A: Ridge strength at K = 3, per-mouse difference from Ridge 0.1, mean +- SEM across mice
        s1 = pm[(pm.K == 3) & (pm.stim == 'stim')]
        s1 = paired_delta(s1, s1.ridge == BEST_RIDGE)
        for cohort, c in COH.items():
            g = s1[s1.cohort == cohort].groupby('ridge').delta.agg(['mean', 'sem'])
            ax[0].errorbar(g.index, 1000 * g['mean'], 1000 * g['sem'], color=c, marker='o', label=cohort)
        ax[0].axhline(0, color='#8a8985', lw=0.8)
        ax[0].set(xscale='log', xlabel='Ridge strength (inverse prior variance)',
                  ylabel=f'Δ held-out LL vs Ridge {BEST_RIDGE} (mbits/trial)', title='Ridge strength, K = 3')
        ax[0].legend(frameon=False)

        # B: K sweep at the best Ridge: gain over K = 1 (a plain GLM); per-mouse fits (solid), global only (dashed)
        s2 = pm[(pm.stim == 'stim') & (pm.ridge == BEST_RIDGE)]
        for cohort, c in COH.items():
            for col, ls in [('bits', '-'), ('bits_global', '--')]:
                d = paired_delta(s2[s2.cohort == cohort], s2[s2.cohort == cohort].K == 1, col)
                g = d.groupby('K').delta.agg(['mean', 'sem'])
                ax[1].errorbar(g.index, 1000 * g['mean'], 1000 * g['sem'], color=c, ls=ls, marker='o',
                               label=f'{cohort}, {"per-mouse" if col == "bits" else "global only"}')
        ax[1].set(xlabel='number of states K', ylabel='Δ held-out LL vs K = 1 (mbits/trial)',
                  title=f'Number of states (Ridge {BEST_RIDGE})', xticks=[1, 2, 3, 4])
        ax[1].legend(frameon=False, fontsize=8)

        # C: contrast transform at K = 3: tanh minus raw, one dot per mouse
        s3 = pm[(pm.K == 3) & (pm.ridge == BEST_RIDGE)]
        s3 = paired_delta(s3, s3.stim == 'stim')
        s3 = s3[s3.stim == 'stim_tanh']
        for i, (cohort, c) in enumerate(COH.items()):
            d = 1000 * s3[s3.cohort == cohort].delta
            ax[2].scatter(np.full(len(d), i), d, color=c,
                          s=20, alpha=0.6, edgecolor='white')
            ax[2].errorbar(i + 0.25, d.mean(), d.sem(), color='#0b0b0b', marker='o')
        ax[2].axhline(0, color='#8a8985', lw=0.8)
        ax[2].set(xticks=[0, 1], xticklabels=list(COH), ylabel='Δ held-out LL, tanh − raw (mbits/trial)',
                  title='Contrast transform, K = 3')

        # D, E: global K = 3 weights and transitions at the best Ridge, every fold (states sorted by stim weight)
        units = [json.loads(f.read_text()) for f in CV_DIR.glob(f'K3_ridge{BEST_RIDGE}_flat_stim_fold*.json')]
        labels = ['stim', 'prev choice', 'wsls', 'block', 'intercept']
        colors = ['#2a78d6', '#4a3aa7', '#1baf7a']
        for u in units:
            W = np.vstack([np.array(u['global_coef']), np.array(u['global_intercept'])[None]])
            for k, idx in enumerate(np.argsort(-W[0])):
                ax[3].plot(range(5), W[:, idx], 'o-', color=colors[k], alpha=0.6, lw=1,
                           label=['state 1 (highest stim)', 'state 2', 'state 3'][k] if u is units[0] else None)
        ax[3].axhline(0, color='#8a8985', lw=0.8)
        ax[3].set_xticks(range(5), labels)
        ax[3].set(ylabel='weight', title=f'Global K = 3 weights, {len(units)} folds')
        ax[3].legend(frameon=False, fontsize=8)
        A = np.mean([np.array(u['global_transition'])[np.ix_(np.argsort(-np.array(u['global_coef'])[0]),
                                                             np.argsort(-np.array(u['global_coef'])[0]))]
                     for u in units], axis=0)
        ax[4].imshow(A, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax[4].text(j, i, f'{A[i, j]:.3f}', ha='center', va='center', color='white' if A[i, j] > 0.5 else 'black')
        ax[4].set(xticks=range(3), yticks=range(3), xlabel='to state', ylabel='from state',
                  title='Global transitions (mean over folds)')
        for a in ax[:4]:
            a.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / 'exploratory-analyses' / '003_model_selection.png', dpi=120)
