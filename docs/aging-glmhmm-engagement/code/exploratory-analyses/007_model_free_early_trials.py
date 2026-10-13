"""Model-free check (user suggestion 2026-10-12): accuracy and reaction time on easy trials at the start of sessions.

If older mice err on easy, high-contrast trials early in the session, when an engaged mouse should be near perfect
and before any block prior exists (the first ~90 trials are an unbiased 50:50 block), the age difference is real
behaviour and not created by the GLM-HMM or the block input. Reaction times in the same window separate slowness
(slow but correct) from inattention (errors, with slow or fast responses).
Exploration mice only. Easy = |contrast| >= 0.25 (0.25 and 1.0); reaction time = firstMovement - stimOn;
response time = response - stimOn (both in s).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
OUT = Path(__file__).resolve().parents[1]
WINDOWS = [20, 50, 90]          # first N trials of each session
MAIN_WINDOW = 50
COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}

dm = pd.read_parquet(OUT / 'artifacts' / '001_design_matrix.parquet')
dm = dm[dm.split == 'exploration']
times = pd.read_parquet(DATA / 'trials.parquet', columns=['session_id', 'trial_id', 'stimOn_times',
                                                          'firstMovement_times', 'response_times'])
t = dm.merge(times, on=['session_id', 'trial_id'])
t['reaction_time'] = t.firstMovement_times - t.stimOn_times
t['response_time'] = t.response_times - t.stimOn_times
t['easy'] = t.stim.abs() >= 0.25
t['full'] = t.stim.abs() == 1
print('fraction of trials with probabilityLeft = 0.5 in the first 90 trials:',
      round((t[t.trial_id < 90].probabilityLeft == 0.5).mean(), 3))

rows = []
for w in WINDOWS:
    e = t[(t.trial_id < w) & t.easy]
    g = e.groupby('subject_id')
    m = pd.DataFrame({'cohort': g.cohort.first(), 'age': g.age_at_session_days.mean(), 'n_easy': g.size(),
                      'acc_easy': g.correct.mean(),
                      'acc_full': e[e.full].groupby('subject_id').correct.mean(),
                      'rt_easy': g.reaction_time.median(),
                      'rt_correct': e[e.correct == 1].groupby('subject_id').reaction_time.median(),
                      'rt_error': e[e.correct == 0].groupby('subject_id').reaction_time.median()})
    rows.append(m.assign(window=w).reset_index())
per_mouse = pd.concat(rows)
per_mouse.to_csv(OUT / 'artifacts' / '007_early_trials_per_mouse.csv', index=False)

stats = []
for w, m in per_mouse.groupby('window'):
    life, bwm = m[m.cohort == 'lifespan'], m[m.cohort == 'bwm']
    for col in ['acc_easy', 'acc_full', 'rt_easy', 'rt_correct']:
        u = mannwhitneyu(life[col].dropna(), bwm[col].dropna())
        stats.append(dict(window=w, metric=col, lifespan_median=life[col].median(), bwm_median=bwm[col].median(),
                          auc=u.statistic / (life[col].notna().sum() * bwm[col].notna().sum()), p=u.pvalue,
                          rho_within_lifespan=spearmanr(life.age, life[col], nan_policy='omit').statistic,
                          rho_within_bwm=spearmanr(bwm.age, bwm[col], nan_policy='omit').statistic,
                          median_n_easy_lifespan=life.n_easy.median()))
stats = pd.DataFrame(stats)
print(stats.round(3).to_string())

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
m = per_mouse[per_mouse.window == MAIN_WINDOW]
for cohort, c in COH.items():
    mm = m[m.cohort == cohort]
    ax[0].scatter(mm.age, mm.acc_easy, color=c, s=40, edgecolor='white', label=cohort)
    ax[2].scatter(mm.age, mm.rt_easy, color=c, s=40, edgecolor='white', label=cohort)
ax[0].set(xlabel='mouse age (days)', ylabel='accuracy, |contrast| ≥ 0.25',
          title=f'Easy-trial accuracy, first {MAIN_WINDOW} trials of each session')
ax[2].set(xlabel='mouse age (days)', ylabel='median reaction time (s)', yscale='log',
          title=f'Reaction time on easy trials, first {MAIN_WINDOW} trials')
ax[0].legend(frameon=False)

# accuracy by window, mean +- SEM across mice; dashed = 100% contrast only
for cohort, c in COH.items():
    for col, ls in [('acc_easy', '-'), ('acc_full', '--')]:
        g = per_mouse[per_mouse.cohort == cohort].groupby('window')[col].agg(['mean', 'sem'])
        ax[1].errorbar(g.index, g['mean'], g['sem'], color=c, ls=ls, marker='o',
                       label=f'{cohort}, {"≥ 0.25" if col == "acc_easy" else "100% only"}')
ax[1].set(xlabel='first N trials of session', ylabel='accuracy', xticks=WINDOWS, ylim=(0.6, 1.02),
          title='Easy-trial accuracy by window')
ax[1].legend(frameon=False, fontsize=8)

# reaction-time distributions on easy trials in the first 90 trials: correct vs error, per cohort (pooled trials)
e = t[(t.trial_id < 90) & t.easy & (t.reaction_time > 0)]
bins = np.logspace(-2, 1.5, 40)
for cohort, c in COH.items():
    for outcome, ls in [(1, '-'), (0, '--')]:
        r = e[(e.cohort == cohort) & (e.correct == outcome)].reaction_time
        ax[3].hist(r, bins, density=True, histtype='step', color=c, ls=ls, lw=1.8,
                   label=f'{cohort} {"correct" if outcome else "error"} (n={len(r)})')
ax[3].set(xscale='log', xlabel='reaction time (s)', ylabel='density',
          title='Easy trials, first 90 trials: correct vs error')
ax[3].legend(frameon=False, fontsize=8)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'exploratory-analyses' / '007_model_free_early_trials.png', dpi=110)
