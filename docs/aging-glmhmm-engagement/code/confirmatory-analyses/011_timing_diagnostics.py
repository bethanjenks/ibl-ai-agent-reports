"""POST HOC (user request 2026-10-12): are the lifespan cohort's "premature" errors a rig / logging artefact?

Checks, per session, in three groups: BWM other labs, BWM churchlandlab (same lab as 18/19 lifespan mice, younger),
and lifespan:
  stim_minus_trigger  stimOn - goCueTrigger (s). In ephysChoiceWorld the stimulus and go cue are triggered together,
                      so this is the screen/photodiode latency. A shifted or variable value suggests stimOn logging
                      problems (another lab's premature responses were a stimOn logging error).
  wait_before_stim    stimOn - intervals_0 (trial start = start of quiescent period): the quiescence actually imposed
  premove             fraction of trials with firstMovement < stimOn
  premove_trigger     fraction with firstMovement < goCueTrigger (independent of the stimOn detection)
  wheel_moved_pre     fraction of trials whose wheel position range in [stimOn - 0.4 s, stimOn] exceeds 2 degrees
                      (approximate; the rig's own quiescence threshold is not in the package)
The quiescencePeriod and stimOnTrigger datasets exist only on OpenAlyx and are not used here.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATA = resolve_dataset_dir('ibl_aging_zang2025')
OUT = Path(__file__).resolve().parents[1]
PRE_WINDOW = 0.4                       # s before stimOn (shortest IBL quiescence period)
MOVE_THRESHOLD = np.deg2rad(2)         # rad
GROUP_COLORS = {'BWM other labs': '#2a78d6', 'BWM churchlandlab': '#1baf7a', 'lifespan': '#eb6834'}

trials = pd.read_parquet(DATA / 'metadata' / 'trials.parquet')
sessions = pd.read_parquet(DATA / 'metadata' / 'sessions.parquet')
t = trials[trials.choice != 0].merge(sessions[['session_id', 'source_group', 'lab', 'task_protocol']], on='session_id')
t['group'] = np.where(t.source_group == 'lifespan', 'lifespan',
                      np.where(t.lab == 'churchlandlab', 'BWM churchlandlab', 'BWM other labs'))
t['stim_minus_trigger'] = t.stimOn_times - t.goCueTrigger_times
t['wait_before_stim'] = t.stimOn_times - t.intervals_0
t['premove'] = t.firstMovement_times < t.stimOn_times
t['premove_trigger'] = t.firstMovement_times < t.goCueTrigger_times


def wheel_moved(session_id, stim_on):
    """For each stimOn time, True if the wheel position range in [stimOn - PRE_WINDOW, stimOn] > MOVE_THRESHOLD."""
    w = pd.read_parquet(DATA / 'wheel' / f'{session_id}.parquet')
    tt, pos = w.time.values, w.position.values
    lo, hi = np.searchsorted(tt, stim_on - PRE_WINDOW), np.searchsorted(tt, stim_on, side='right')
    # position just before the window counts as the window's starting position
    return np.array([np.ptp(pos[max(a - 1, 0):b]) > MOVE_THRESHOLD if b > 0 else np.nan for a, b in zip(lo, hi)])


t['wheel_moved_pre'] = np.concatenate([wheel_moved(s, g.stimOn_times.values)
                                       for s, g in t.groupby('session_id', sort=False)])
cols = ['stim_minus_trigger', 'wait_before_stim', 'premove', 'premove_trigger', 'wheel_moved_pre']
sess = t.groupby(['session_id', 'group', 'task_protocol'])[cols].median(numeric_only=True)
sess[['premove', 'premove_trigger', 'wheel_moved_pre']] = t.groupby(['session_id', 'group', 'task_protocol'])[
    ['premove', 'premove_trigger', 'wheel_moved_pre']].mean()
sess = sess.reset_index()
sess['stim_minus_trigger_iqr'] = sess.session_id.map(
    t.groupby('session_id').stim_minus_trigger.agg(lambda x: x.quantile(0.75) - x.quantile(0.25)))
sess.to_csv(OUT / 'artifacts' / '011_timing_per_session.csv', index=False)

summary = sess.groupby('group')[cols + ['stim_minus_trigger_iqr']].median().round(4)
summary['sessions'] = sess.groupby('group').size()
print(summary.T.to_string())
print('\nprotocol versions by group (sessions):')
print(pd.crosstab(sess.task_protocol.str.replace('_iblrig_tasks_ephysChoiceWorld', 'v'), sess.group).to_string())
v64 = sess[sess.task_protocol.str.contains('6.4')]
print('\nsame protocol family (6.4.x) only:')
print(v64.groupby('group')[cols].median().round(4).T.to_string())

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
panels = [('stim_minus_trigger', 'stimOn − goCueTrigger, per-session median (s)', np.linspace(-0.05, 0.25, 61)),
          ('wait_before_stim', 'stimOn − trial start, per-session median (s)', np.linspace(0, 3, 61)),
          ('premove', 'fraction of trials with firstMovement < stimOn', np.linspace(0, 0.8, 41)),
          ('wheel_moved_pre', f'fraction with wheel > 2° in {PRE_WINDOW} s before stimOn', np.linspace(0, 0.8, 41))]
for a, (col, xlabel, bins) in zip(ax, panels):
    for g, c in GROUP_COLORS.items():
        v = sess[sess.group == g][col].dropna()
        a.hist(v, bins, density=True, histtype='step', lw=2, color=c, label=f'{g} ({len(v)} sessions)')
    a.set(xlabel=xlabel, ylabel='density')
    a.spines[['top', 'right']].set_visible(False)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('POST HOC timing diagnostics: stimulus logging and pre-stimulus wheel movement by group', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'confirmatory-analyses' / '011_timing_diagnostics.png', dpi=110)
