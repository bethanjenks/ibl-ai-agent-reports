"""NeMoS GLM-HMM smoke test on 12 exploration mice (all 6 lifespan + 6 bwm).

(a) Parameter recovery. Choices are simulated from a known 3-state GLM-HMM (one engaged, two biased states) using
    each mouse's real stimulus and block sequence; history inputs are regenerated from the simulated choices.
    We then fit with the planned procedure: a global fit on all mice (HMM reset at each session start), then a
    per-mouse fit started from the global parameters. Checks: recovered weights and transitions vs truth, and
    per-mouse engaged-state occupancy vs the true occupancy.
(b) The same procedure on the real choices of the same mice, with wall-clock times, to estimate full run time.

Engaged state = the state with the largest stimulus weight. Occupancy = fraction of trials whose smoothed
posterior for that state is > 0.5. Ridge strength 0.5 = Gaussian prior variance 2 on weights (Ashwood-like);
Dirichlet alpha 2 on transitions. Both are placeholders until the grid in step 5.
"""
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import nemos as nmo
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
FEATURES = ['stim', 'prev_choice', 'wsls', 'block']
K = 3
RIDGE = 0.5
DIRICHLET_ALPHA = 2.0
N_BWM_MICE = 6
SEED = 20261012
rng = np.random.default_rng(SEED)

# Ground truth for simulation: weights (features x states), intercepts, transitions. Engaged state first.
TRUE_W = np.array([[6.0, 1.0, 1.0],      # stim
                   [0.3, 0.5, 0.5],      # prev_choice
                   [0.2, 0.0, 0.0],      # wsls
                   [0.5, 0.3, 0.3]])     # block
TRUE_B = np.array([0.0, -2.0, 2.0])
TRUE_A = np.array([[0.985, 0.0075, 0.0075],
                   [0.03, 0.96, 0.01],
                   [0.03, 0.01, 0.96]])
STATE_NAMES = ['engaged', 'biased L', 'biased R']

dm = pd.read_parquet(OUT / 'artifacts' / '001_design_matrix.parquet')
dm = dm[dm.split == 'exploration']
mice_all = dm.groupby('subject_id').cohort.first()
lifespan_mice = list(mice_all[mice_all == 'lifespan'].index)
bwm_mice = list(rng.choice(mice_all[mice_all == 'bwm'].index, N_BWM_MICE, replace=False))
dm = dm[dm.subject_id.isin(lifespan_mice + bwm_mice)].sort_values(['subject_id', 'date', 'session_id', 'trial_id'])


def simulate_session(stim, block):
    """Simulate one session from the true GLM-HMM. stim, block: (n_trials,). History inputs are rebuilt from the
    simulated choices. Returns X (n_trials, 4), y (n_trials,), true states (n_trials,)."""
    n = len(stim)
    X, y, z = np.zeros((n, 4)), np.zeros(n, int), np.zeros(n, int)
    state, prev_c, prev_r = rng.choice(K, p=[0.8, 0.1, 0.1]), 0.0, 0.0
    for i in range(n):
        if i > 0:
            state = rng.choice(K, p=TRUE_A[state])
        X[i] = [stim[i], prev_c, prev_c * prev_r, block[i]]
        y[i] = rng.random() < 1 / (1 + np.exp(-(X[i] @ TRUE_W[:, state] + TRUE_B[state])))
        prev_c = 2 * y[i] - 1
        prev_r = 1.0 if (stim[i] > 0) == y[i] or (stim[i] == 0 and rng.random() < 0.5) else -1.0
        z[i] = state
    return X, y, z


def model():
    return nmo.glm_hmm.GLMHMM(n_states=K, regularizer_strength=RIDGE, seed=jax.random.PRNGKey(SEED),
                              dirichlet_transition_proba=np.full((K, K), DIRICHLET_ALPHA))


def fit_global_and_mice(X, y, starts, mouse):
    """Global fit on all trials, then per-mouse fits started from the global parameters.
    X (n, 4), y (n,), starts (n,) bool session starts, mouse (n,) mouse ids.
    Returns the global model, a dict mouse -> fitted model, and the time taken (s)."""
    t0 = time.time()
    glob = model().fit(X, y, session_starts=starts)
    init = (glob.coef_, glob.intercept_, glob.scale_, glob.initial_prob_, glob.transition_prob_)
    per_mouse = {m: model().fit(X[mouse == m], y[mouse == m], init_params=init, session_starts=starts[mouse == m])
                 for m in np.unique(mouse)}
    return glob, per_mouse, time.time() - t0


def engaged_occupancy(mdl, X, y, starts):
    """Fraction of trials with smoothed posterior > 0.5 for the engaged (largest stimulus weight) state."""
    post = np.asarray(mdl.smooth_proba(X, y, session_starts=starts))
    return np.mean(post[:, np.argmax(np.asarray(mdl.coef_)[0])] > 0.5), post


# ---------------- (a) parameter recovery on simulated choices ----------------
sim_X, sim_y, sim_z = [], [], []
for _, s in dm.groupby('session_id', sort=False):
    X_s, y_s, z_s = simulate_session(s.stim.values, s.block.values)
    sim_X.append(X_s); sim_y.append(y_s); sim_z.append(z_s)
sim_X, sim_y, sim_z = np.concatenate(sim_X), np.concatenate(sim_y), np.concatenate(sim_z)
starts = (dm.session_id != dm.session_id.shift()).values
mouse = dm.subject_id.values

sim_glob, sim_mice, t_sim = fit_global_and_mice(sim_X, sim_y, starts, mouse)
W = np.vstack([np.asarray(sim_glob.coef_), np.asarray(sim_glob.intercept_)[None]])   # (5, K)
TRUE_WB = np.vstack([TRUE_W, TRUE_B[None]])
perm = min(itertools.permutations(range(K)), key=lambda p: np.sum((W[:, list(p)] - TRUE_WB) ** 2))
W = W[:, list(perm)]
A = np.asarray(sim_glob.transition_prob_)[np.ix_(perm, perm)]
print(f'simulation: {len(sim_y)} trials, fit time {t_sim:.0f} s (global + {len(sim_mice)} mice)')
print('recovered weights (rows stim, prev_choice, wsls, block, intercept):\n', W.round(2))
print('recovered transitions:\n', A.round(3))

occ = []
for m, mdl in sim_mice.items():
    sel = mouse == m
    est, post = engaged_occupancy(mdl, sim_X[sel], sim_y[sel], starts[sel])
    accuracy = np.mean((post.argmax(1) == np.argmax(np.asarray(mdl.coef_)[0])) == (sim_z[sel] == 0))
    occ.append(dict(subject_id=m, true=np.mean(sim_z[sel] == 0), est=est, engaged_label_accuracy=accuracy,
                    n_trials=sel.sum(), cohort=mice_all[m]))
occ = pd.DataFrame(occ)
print(occ.round(3))

# ---------------- (b) the same procedure on real choices ----------------
X_real, y_real = dm[FEATURES].values.astype(float), dm.y.values
real_glob, real_mice, t_real = fit_global_and_mice(X_real, y_real, starts, mouse)
order = np.argsort(-np.asarray(real_glob.coef_)[0])                                    # engaged first
W_real = np.vstack([np.asarray(real_glob.coef_), np.asarray(real_glob.intercept_)[None]])[:, order]
print(f'real: {len(y_real)} trials, fit time {t_real:.0f} s')
print('real global weights (engaged first):\n', W_real.round(2))
print('real global transitions:\n', np.asarray(real_glob.transition_prob_)[np.ix_(order, order)].round(3))
real_occ = pd.DataFrame([dict(subject_id=m, cohort=mice_all[m], n_trials=(mouse == m).sum(),
                              occ=engaged_occupancy(mdl, X_real[mouse == m], y_real[mouse == m], starts[mouse == m])[0])
                         for m, mdl in real_mice.items()])
print(real_occ.round(3))
real_occ.to_csv(OUT / 'artifacts' / '002_real_occupancy_pilot.csv', index=False)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
labels = ['stim', 'prev choice', 'wsls', 'block', 'intercept']
x = np.arange(len(labels))
colors = ['#2a78d6', '#4a3aa7', '#1baf7a']
for k in range(K):
    ax[0].plot(x, TRUE_WB[:, k], 'o', color=colors[k], ms=9, mfc='none', mew=2)
    ax[0].plot(x, W[:, k], 'x', color=colors[k], ms=8, mew=2, label=STATE_NAMES[k])
ax[0].set_xticks(x, labels)
ax[0].set(ylabel='weight', title='Simulation: true (○) vs recovered (×) global weights')
ax[0].legend(frameon=False)

for cohort, c in [('bwm', '#2a78d6'), ('lifespan', '#eb6834')]:
    o = occ[occ.cohort == cohort]
    ax[1].scatter(o.true, o.est, color=c, s=40, edgecolor='white', label=cohort)
lim = [occ[['true', 'est']].min().min() - 0.05, 1]
ax[1].plot(lim, lim, color='#8a8985', lw=1, ls=':')
ax[1].set(xlabel='true engaged occupancy', ylabel='estimated (per-mouse fit)',
          title='Simulation: per-mouse occupancy recovery')
ax[1].legend(frameon=False)

for k in range(K):
    ax[2].plot(x, W_real[:, k], 'o-', color=colors[k], ms=7, lw=1.5, label=['engaged', 'state 2', 'state 3'][k])
ax[2].axhline(0, color='#8a8985', lw=0.8)
ax[2].set_xticks(x, labels)
ax[2].set(ylabel='weight', title=f'Real data: global weights ({len(real_mice)} mice)')
ax[2].legend(frameon=False)

# Example real session: smoothed posteriors for one lifespan mouse's first session
m = lifespan_mice[0]
sel = mouse == m
_, post = engaged_occupancy(real_mice[m], X_real[sel], y_real[sel], starts[sel])
first = np.cumsum(starts[sel]) == 1
order_m = np.argsort(-np.asarray(real_mice[m].coef_)[0])
for k, idx in enumerate(order_m):
    ax[3].plot(post[first, idx], color=colors[k], lw=1.2, label=['engaged', 'state 2', 'state 3'][k])
err = (dm[sel][first].correct.values == 0)
ax[3].plot(np.where(err)[0], np.full(err.sum(), 1.05), '|', color='#0b0b0b', ms=6, label='error')
ax[3].set(xlabel='trial', ylabel='posterior P(state)', ylim=(0, 1.1), title=f'Real example: {m}, session 1')
ax[3].legend(frameon=False, fontsize=8, ncol=4, loc='lower left')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '002_nemos_smoke_test.png', dpi=120)
