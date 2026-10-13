"""Choose a movement-safe post-stimulus window (bwm exploration mice only, so the choice is blind to age).

The paper's post window [0.16, 0.26] s mostly follows first movement (median 0.14 s after stimOn). Here, on
engaged bwm trials (P > 0.8):
- left: population FF and FR time courses for trials grouped by first-movement latency (< 0.16, 0.16–0.3, > 0.3 s),
  equal trial counts per group, showing which part of the post-stimulus FF change precedes movement;
- right: candidate movement-safe quench = FF(window ending at E) − FF(window ending at 0) using only trials with
  first movement after E, for E in END_GRID; mean quench across neurons and trials left per session.
Candidate rule: the window end E with the clearest pre-movement quenching that leaves enough trials in lifespan-like
sessions (~280 engaged trials, i.e. trial fraction × 280 >= ~80).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quench_common import PROJECT, WINDOW_ENDS, WIDTH, load_trials, load_units, load_spikes, window_counts, \
    mean_subtracted_ff

END_GRID = [0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.26]
LATENCY_GROUPS = {'< 0.16 s': (-np.inf, 0.16), '0.16–0.3 s': (0.16, 0.3), '> 0.3 s': (0.3, np.inf)}
N_PER_GROUP = 40
GROUP_COLORS = ['#d6552a', '#888888', '#2a78d6']
rng = np.random.default_rng(0)

units = load_units()
units = units[(units.split == 'exploration') & (units.cohort == 'bwm')]
trials = load_trials()
trials = trials[trials.session_id.isin(units.session_id) & (trials.p_engaged > 0.8)].sort_values(['session_id', 'trial_id'])
trials['mv'] = trials.firstMovement_times - trials.stimOn_times

ff_groups, fr_groups, rows = {g: [] for g in LATENCY_GROUPS}, {g: [] for g in LATENCY_GROUPS}, []
for session_id, st in trials.groupby('session_id'):
    su = pd.concat([u for _, u in units[units.session_id == session_id].groupby('recording_id')])
    counts = np.concatenate([window_counts(*load_spikes(rec), u.unit_index.values, st.stimOn_times.values)
                             for rec, u in su.groupby('recording_id')], axis=1)
    contrasts, mv = st.signed_contrast.values, st.mv.values
    for g, (lo, hi) in LATENCY_GROUPS.items():
        idx = np.flatnonzero((mv > lo) & (mv <= hi))
        if len(idx) >= N_PER_GROUP:
            d = rng.choice(idx, N_PER_GROUP, replace=False)
            ff_groups[g].append(mean_subtracted_ff(counts[d], contrasts[d]))
            fr_groups[g].append(counts[d].mean(0) / WIDTH)
    for end in END_GRID:
        idx = np.flatnonzero(mv > end)
        if len(idx) < 30:
            continue
        ff = mean_subtracted_ff(counts[idx], contrasts[idx])
        k_end, k_pre = np.searchsorted(WINDOW_ENDS, end), np.searchsorted(WINDOW_ENDS, 0.0)
        rows.append({'session_id': session_id, 'end': end, 'frac_trials': len(idx) / len(st),
                     'quench_mean': np.nanmean(ff[:, k_end] - ff[:, k_pre]),
                     'quench_median': np.nanmedian(ff[:, k_end] - ff[:, k_pre])})
grid = pd.DataFrame(rows)
summary = grid.groupby('end').agg(sessions=('session_id', 'size'), frac_trials=('frac_trials', 'median'),
                                  quench_mean=('quench_mean', 'median'), quench_median=('quench_median', 'median'))
summary['trials_if_280_engaged'] = (summary.frac_trials * 280).round()
print(summary.round(3))

fig, ax = plt.subplots(1, 4, figsize=(22, 5))
for (g, ffs), color in zip(ff_groups.items(), GROUP_COLORS):
    ff_all, fr_all = np.concatenate(ffs), np.concatenate(fr_groups[g])
    ax[0].plot(WINDOW_ENDS, np.nanmean(ff_all, 0), color=color, label=f'move {g} ({len(ffs)} sessions)')
    ax[1].plot(WINDOW_ENDS, np.nanmean(fr_all, 0), color=color, label=f'move {g}')
for a, label in [(ax[0], 'mean-subtracted FF'), (ax[1], 'FR (Hz)')]:
    a.axvspan(0.16, 0.26, color='grey', alpha=0.2, label='paper post window (end 0.26)')
    a.axvline(0, color='k', lw=0.5)
    a.set(xlabel='window end from stimOn (s)', ylabel=f'mean over neurons ({N_PER_GROUP} trials per group)',
          title=f'bwm engaged trials by first-movement latency: {label}')
    a.legend(fontsize=8)
ax[2].plot(summary.index, summary.quench_mean, 'o-', label='mean over neurons')
ax[2].plot(summary.index, summary.quench_median, 's--', label='median over neurons')
ax[2].axhline(0, color='grey')
ax[2].set(xlabel='post window end E (s); trials with movement after E', ylabel='session median of quench',
          title='Movement-safe quench vs window end')
ax[2].legend()
ax[3].plot(summary.index, summary.frac_trials, 'o-', color='k')
ax[3].set(xlabel='post window end E (s)', ylabel='median fraction of engaged trials kept', ylim=(0, 1),
          title='Trials left after excluding movement before E')
fig.tight_layout()
fig.savefig(PROJECT / 'exploratory-analyses' / '003_movement_safe_window.png', dpi=110)
