"""Metric-definition figure: FR, mean-subtracted FF and quench on engaged vs disengaged trials, two sessions.

Picks one bwm and one lifespan exploration session with >= 50 disengaged trials (P(engaged) < 0.2) and the most
paper-ROI neurons. Engaged trials (P > 0.8) are randomly subsampled to the number of disengaged trials (N_REPEATS
draws, metrics averaged), because FF estimates depend on trial count.
Rows: sessions. Columns: P(engaged) over the session; raster of one example neuron sorted by state; population
FR time course; population FF time course; per-neuron quench, engaged vs disengaged.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quench_common import (PROJECT, WINDOW_ENDS, WIDTH, PRE_END, POST_END, load_trials, load_units, load_spikes,
                           window_counts, mean_subtracted_ff, state_metrics)

THETA, MIN_DISENGAGED, N_REPEATS = 0.8, 50, 20
STATE_COLORS = {'engaged': '#1baf7a', 'disengaged': '#9b59b6'}
rng = np.random.default_rng(0)

units = load_units()
units = units[units.split == 'exploration']
trials = load_trials()
trials['state'] = np.select([trials.p_engaged > THETA, trials.p_engaged < 1 - THETA], ['engaged', 'disengaged'], '')
n_dis = trials[trials.state == 'disengaged'].groupby('session_id').size()
candidates = units[units.session_id.isin(n_dis[n_dis >= MIN_DISENGAGED].index)]
picks = candidates.groupby(['cohort', 'session_id']).size().groupby('cohort').idxmax().str[1]
print('sessions:', picks.to_dict(), 'disengaged trials:', n_dis[picks].to_dict())

fig, ax = plt.subplots(2, 5, figsize=(24, 9))
for row, (cohort, session_id) in enumerate(picks.items()):
    st = trials[trials.session_id == session_id].sort_values('trial_id')
    su = units[units.session_id == session_id]
    eng, dis = st[st.state == 'engaged'], st[st.state == 'disengaged']
    n = len(dis)
    # counts for every recording of the session, units concatenated
    counts = {}
    for name, sub in [('engaged', eng), ('disengaged', dis)]:
        counts[name] = np.concatenate([window_counts(*load_spikes(rec), u.unit_index.values, sub.stimOn_times.values)
                                       for rec, u in su.groupby('recording_id')], axis=1)
    su = pd.concat([u for _, u in su.groupby('recording_id')])  # same unit order as counts
    # engaged: average over subsamples of n trials; disengaged: all n
    draws = [rng.choice(len(eng), n, replace=False) for _ in range(N_REPEATS)]
    ff = {'engaged': np.mean([mean_subtracted_ff(counts['engaged'][d], eng.signed_contrast.values[d]) for d in draws], 0),
          'disengaged': mean_subtracted_ff(counts['disengaged'], dis.signed_contrast.values)}
    fr = {'engaged': counts['engaged'].mean(0) / WIDTH, 'disengaged': counts['disengaged'].mean(0) / WIDTH}
    quench = {'engaged': np.mean([state_metrics(counts['engaged'][d], eng.signed_contrast.values[d])['quench']
                                  for d in draws], 0),
              'disengaged': state_metrics(counts['disengaged'], dis.signed_contrast.values)['quench']}

    title = f'{cohort} {st.subject_id.iloc[0] if "subject_id" in st else ""} {su.age_at_session_days.iloc[0]:.0f} d'
    a = ax[row, 0]
    a.plot(st.trial_rank, st.p_engaged, color='k', lw=0.6)
    a.axhline(THETA, color=STATE_COLORS['engaged'], ls='--'); a.axhline(1 - THETA, color=STATE_COLORS['disengaged'], ls='--')
    a.set(xlabel='trial', ylabel='P(engaged)', title=f'{title}: {len(eng)} engaged, {n} disengaged')

    # example neuron: highest pre-stimulus rate among visual/motor cortex units if present
    roi_pref = su.roi.isin(['VISp+pm', 'MOs']).values
    k = np.argmax((fr['engaged'][:, 15]) * (roi_pref + 0.01))
    a = ax[row, 1]
    spike_times, spike_units = load_spikes(su.recording_id.iloc[k])
    unit_spikes = spike_times[spike_units == su.unit_index.iloc[k]]
    y = 0
    for name, sub in [('disengaged', dis), ('engaged', eng.iloc[draws[0]])]:
        for t0 in sub.sort_values('signed_contrast').stimOn_times.values:
            s = unit_spikes[(unit_spikes > t0 - 0.4) & (unit_spikes < t0 + 0.6)] - t0
            a.plot(s, np.full_like(s, y), '|', ms=2, color=STATE_COLORS[name])
            y += 1
    for end, label in [(PRE_END, 'pre'), (POST_END, 'post')]:
        a.axvspan(end - WIDTH, end, color='grey', alpha=0.2)
    a.set(xlabel='time from stimOn (s)', ylabel='trials (sorted by state, then contrast)',
          title=f'{su.roi.iloc[k]} unit {su.unit_index.iloc[k]}: quench eng {quench["engaged"][k]:.2f}, '
                f'dis {quench["disengaged"][k]:.2f}')

    for col, metric, label in [(2, fr, 'FR (Hz), 100 ms window'), (3, ff, 'mean-subtracted FF')]:
        a = ax[row, col]
        for name in metric:
            a.plot(WINDOW_ENDS, np.nanmean(metric[name], 0), color=STATE_COLORS[name], label=name)
        for end in [PRE_END, POST_END]:
            a.axvspan(end - WIDTH, end, color='grey', alpha=0.2)
        a.axvline(0, color='k', lw=0.5)
        a.set(xlabel='window end from stimOn (s)', ylabel=f'mean over {len(su)} neurons', title=label)
        a.legend()
    a = ax[row, 4]
    a.scatter(quench['engaged'], quench['disengaged'], s=8, color='k', alpha=0.5)
    lim = np.nanpercentile(np.r_[quench['engaged'], quench['disengaged']], [1, 99])
    a.plot(lim, lim, color='grey'); a.axhline(0, color='grey', lw=0.5); a.axvline(0, color='grey', lw=0.5)
    d = quench['disengaged'] - quench['engaged']
    a.set(xlim=lim, ylim=lim, xlabel=f'quench, engaged (n = {n} matched)', ylabel='quench, disengaged',
          title=f'median dis − eng = {np.nanmedian(d):.3f}; {np.mean(d > 0):.0%} of neurons above diagonal')
    print(cohort, session_id, 'neurons', len(su), 'median quench eng/dis',
          np.nanmedian(quench['engaged']).round(3), np.nanmedian(quench['disengaged']).round(3))
fig.tight_layout()
fig.savefig(PROJECT / 'exploratory-analyses' / '001_ff_single_session.png', dpi=110)
