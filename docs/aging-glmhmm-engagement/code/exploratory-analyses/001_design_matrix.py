"""Per-trial GLM-HMM inputs and choices for all sessions, plus behavioural diagnostics on exploration mice.

Inputs follow Ashwood et al. (2022): signed stimulus contrast, previous choice, previous choice x previous reward
(win-stay/lose-switch), and an intercept (added by the model). We add the block prior because these sessions use
biased blocks; without it, block-driven bias would be absorbed into "biased" states.

Coding (checked against the data: correct trials with a right-side stimulus all have choice == -1):
  y = 1 for a rightward choice (choice == -1), 0 for leftward (choice == +1). No-go trials (choice == 0, 0.4%)
  are dropped; history inputs are computed over the remaining trials within each session.
  stim       = contrastRight - contrastLeft (in [-1, 1])
  stim_tanh  = tanh(5 * stim) / tanh(5), a compressive alternative (choice to be made in step 5)
  prev_choice= +1 / -1 for the previous rightward / leftward choice; 0 on a session's first trial
  wsls       = prev_choice * previous reward (+1 correct, -1 error): +1 favours repeating a rewarded choice
  block      = +1 right-biased block (probabilityLeft 0.2), -1 left-biased (0.8), 0 unbiased (0.5)
The table covers all mice (confirmation mice are not looked at here); diagnostics use exploration mice only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
OUT = Path(__file__).resolve().parents[1]
POSITION_BIN = 50          # trials per bin for the within-session accuracy profile
COHORT_COLORS = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}
BLOCK_COLORS = {-1: '#4a3aa7', 0: '#8a8985', 1: '#1baf7a'}
BLOCK_NAMES = {-1: 'left block (pL 0.8)', 0: 'unbiased (pL 0.5)', 1: 'right block (pL 0.2)'}

trials = pd.read_parquet(DATA / 'trials.parquet')
sessions = pd.read_parquet(DATA / 'sessions.parquet')
split = pd.read_csv(OUT / 'artifacts' / 'split.csv')

t = trials[trials.choice != 0].sort_values(['session_id', 'trial_id']).copy()
t['y'] = (t.choice == -1).astype(int)
t['stim'] = t.contrastRight.fillna(0) - t.contrastLeft.fillna(0)
t['stim_tanh'] = np.tanh(5 * t.stim) / np.tanh(5)
t['correct'] = (t.feedbackType == 1).astype(int)
g = t.groupby('session_id')
t['prev_choice'] = (2 * g.y.shift(1) - 1).fillna(0)
t['wsls'] = t.prev_choice * (2 * g.correct.shift(1) - 1).fillna(0)
t['block'] = np.select([t.probabilityLeft == 0.2, t.probabilityLeft == 0.8], [1, -1], 0)
t = t.merge(sessions[['session_id', 'subject_id', 'source_group', 'lab', 'age_at_session_days', 'date']],
            on='session_id').rename(columns={'source_group': 'cohort'})
t = t.merge(split[['subject_id', 'split']], on='subject_id')
cols = ['subject_id', 'session_id', 'date', 'trial_id', 'split', 'cohort', 'lab', 'age_at_session_days',
        'y', 'stim', 'stim_tanh', 'prev_choice', 'wsls', 'block', 'correct', 'probabilityLeft']
t[cols].to_parquet(OUT / 'artifacts' / '001_design_matrix.parquet', index=False)
print(f'{len(t)} trials kept ({(trials.choice == 0).sum()} no-go dropped), {t.session_id.nunique()} sessions')

# ---------------- diagnostics: exploration mice only ----------------
e = t[t.split == 'exploration'].copy()
e['abs_stim'] = e.stim.abs()
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))

# A, B: psychometric curves per cohort and block, mean +- SEM across mice (mouse = independent unit)
for a, cohort in zip(ax[:2], ['bwm', 'lifespan']):
    psy = e[e.cohort == cohort].groupby(['block', 'subject_id', 'stim']).y.mean()
    for b, color in BLOCK_COLORS.items():
        m = psy.loc[b].groupby('stim').agg(['mean', 'sem', 'size'])
        a.errorbar(m.index, m['mean'], m['sem'], color=color, marker='o', ms=4, lw=2, capsize=0,
                   label=BLOCK_NAMES[b])
    n = e[e.cohort == cohort].subject_id.nunique()
    a.set(xlabel='signed contrast (right − left)', ylabel='P(rightward choice)', ylim=(0, 1),
          title=f'{cohort}: psychometric by block ({n} mice)')
ax[0].legend(fontsize=8, frameon=False)

# C: accuracy on easy trials (|contrast| >= 0.25) vs trial position: session-end disengagement
e['position'] = (e.trial_id // POSITION_BIN) * POSITION_BIN
easy = e[e.abs_stim >= 0.25]
for cohort, color in COHORT_COLORS.items():
    acc = easy[easy.cohort == cohort].groupby(['position', 'subject_id']).correct.mean().groupby('position')
    m = acc.agg(['mean', 'sem', 'size'])
    m = m[m['size'] >= 3]                                  # need >= 3 mice per position bin
    ax[2].errorbar(m.index, m['mean'], m['sem'], color=color, lw=2, label=cohort)
ax[2].axvline(400, color='#8a8985', ls=':', lw=1)
ax[2].set(xlabel=f'trial number in session ({POSITION_BIN}-trial bins)', ylabel='accuracy, |contrast| ≥ 0.25',
          title='Within-session accuracy on easy trials', ylim=(0.5, 1))
ax[2].legend(frameon=False)

# D: raw lapse proxy per mouse: error rate at 100% contrast, vs age (the phenomenon H3 must explain)
lapse = (e[e.abs_stim == 1].groupby('subject_id')
         .agg(error=('correct', lambda c: 1 - c.mean()), age=('age_at_session_days', 'mean'), cohort=('cohort', 'first')))
for cohort, color in COHORT_COLORS.items():
    m = lapse[lapse.cohort == cohort]
    ax[3].scatter(m.age, m.error, color=color, s=36, edgecolor='white', lw=0.8, label=cohort)
ax[3].set(xlabel='mean age at session (days)', ylabel='error rate at 100% contrast',
          title='Raw lapse proxy per mouse')
ax[3].legend(frameon=False)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '001_design_matrix.png', dpi=120)

print(e.groupby('cohort').agg(trials=('y', 'size'), sessions=('session_id', 'nunique'), mice=('subject_id', 'nunique')))
print('input correlations (exploration):')
print(e[['stim', 'prev_choice', 'wsls', 'block']].corr().round(2))
