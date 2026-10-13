"""Shared GLM-HMM fitting and per-mouse engagement metrics, so exploration and confirmation compute them identically.

Settings fixed in step 5 (003_model_selection): Ridge 0.1, raw signed contrast, no transition prior (NeMoS 0.2.9
ignores it). K = 2 since step 6b (user choice 2026-10-12): with K = 3 the model alternates between solutions in which a
biased mode is or is not stimulus-driven, so engaged occupancy was unreliable. With per_mouse_label=True the engaged
state of each mouse is the state with the larger stimulus weight in that mouse's own fit.
"""
import os
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
import pandas as pd
import jax
import nemos as nmo
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
FEATURES = ['stim', 'prev_choice', 'wsls', 'block']
K = 2
RIDGE = 0.1
N_RESTARTS = 3
ENGAGED_STIM_MIN = 5.0
SEED = 20261012


def load_design(split):
    """Trials of one split ('exploration' or 'confirmation'), sorted by mouse, date, session, trial.
    Returns the DataFrame and a boolean array (n_trials,) marking session starts."""
    dm = pd.read_parquet(OUT / 'artifacts' / '001_design_matrix.parquet')
    dm = dm[dm.split == split].sort_values(['subject_id', 'date', 'session_id', 'trial_id']).reset_index(drop=True)
    return dm, (dm.session_id != dm.session_id.shift()).values


def make_model(n_states=K, seed=SEED):
    return nmo.glm_hmm.GLMHMM(n_states=n_states, regularizer_strength=RIDGE, seed=jax.random.PRNGKey(seed))


def fit_global(X, y, starts, n_states=K, seeds=None):
    """Global fit on all mice; best of several random restarts by training log-likelihood. Returns the model."""
    seeds = [SEED + r for r in range(N_RESTARTS)] if seeds is None else seeds
    fits = [make_model(n_states, s).fit(X, y, session_starts=starts) for s in seeds]
    return fits[int(np.argmax([float(f.score(X, y, session_starts=starts)) for f in fits]))]


def fit_mouse(glob, X, y, starts):
    """Per-mouse fit started from the global parameters. Starting from the global fit keeps the state order, so
    state k in the mouse model corresponds to state k in the global model (checked in step 6)."""
    init = (glob.coef_, glob.intercept_, glob.scale_, glob.initial_prob_, glob.transition_prob_)
    return make_model(len(glob.intercept_)).fit(X, y, init_params=init, session_starts=starts)


def engaged_states(glob):
    """Indices of global states with stimulus weight > ENGAGED_STIM_MIN."""
    return np.where(np.asarray(glob.coef_)[0] > ENGAGED_STIM_MIN)[0]


def fit_all_mice(glob, dm, starts):
    """Per-mouse fits for every mouse in dm. Returns dict mouse -> model and per-trial posteriors (n_trials, K)."""
    X, y = dm[FEATURES].values.astype(float), dm.y.values.astype(float)
    models, post = {}, np.zeros((len(dm), len(glob.intercept_)))
    for m, idx in dm.groupby('subject_id').indices.items():
        models[m] = fit_mouse(glob, X[idx], y[idx], starts[idx])
        post[idx] = np.asarray(models[m].smooth_proba(X[idx], y[idx], session_starts=starts[idx]))
    return models, post


def run_lengths(mask):
    """Lengths of runs of True in a 1-D boolean array."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.where(edges == -1)[0] - np.where(edges == 1)[0]


def mouse_metrics(dm, post, models, engaged, per_mouse_label=False):
    """Per-mouse engagement metrics.

    dm: trials (n_trials rows); post: (n_trials, K) smoothed state posteriors from the per-mouse fits; models: mouse ->
    model; engaged: engaged state indices from the global fit, used unless per_mouse_label (then each mouse's engaged
    state is its own highest-stimulus-weight state). p_engaged = summed posterior of engaged states; engaged if > 0.5.
      occupancy           fraction of engaged trials (H1)
      soft_occupancy      mean p_engaged (no threshold)
      disengaged_bout     mean length (trials) of runs of disengaged trials, pooled over sessions (H2)
      switches_per_100    engaged <-> disengaged switches per 100 trials, within sessions (H2)
      engaged_stim_weight per-mouse stimulus weight averaged over engaged states, weighted by their occupancy (H3)
      lapse_proxy         error rate on 100% contrast trials (H3 outcome)
    Returns one row per mouse."""
    eng_of = {m: [int(np.argmax(np.asarray(md.coef_)[0]))] if per_mouse_label else list(engaged) for m, md in models.items()}
    p_engaged = np.zeros(len(dm))
    for m, idx in dm.groupby('subject_id', sort=False).indices.items():
        p_engaged[idx] = post[idx][:, eng_of[m]].sum(1)
    d = dm.assign(p_engaged=p_engaged, engaged=p_engaged > 0.5)
    rows = []
    for m, idx in d.groupby('subject_id', sort=False).indices.items():
        g = d.iloc[idx]
        engaged = eng_of[m]
        bouts = np.concatenate([run_lengths(~s.engaged.values) for _, s in g.groupby('session_id')])
        switches = sum(np.abs(np.diff(s.engaged.values.astype(int))).sum() for _, s in g.groupby('session_id'))
        coef = np.asarray(models[m].coef_)[0]
        w = post[idx][:, engaged].sum(0)
        rows.append(dict(subject_id=m, cohort=g.cohort.iloc[0], lab=g.lab.iloc[0],
                         age=g.groupby('session_id').age_at_session_days.first().mean(),
                         n_trials=len(g), n_sessions=g.session_id.nunique(),
                         occupancy=g.engaged.mean(), soft_occupancy=g.p_engaged.mean(),
                         disengaged_bout=bouts.mean() if len(bouts) else np.nan,
                         switches_per_100=100 * switches / len(g),
                         engaged_stim_weight=np.sum(coef[engaged] * w) / w.sum(),
                         disengaged_stim_weight=np.delete(coef, engaged).max() if len(engaged) < len(coef) else np.nan,
                         lapse_proxy=1 - g[g.stim.abs() == 1].correct.mean()))
    return pd.DataFrame(rows)
