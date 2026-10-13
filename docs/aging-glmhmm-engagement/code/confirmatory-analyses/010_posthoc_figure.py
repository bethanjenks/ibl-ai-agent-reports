"""POST HOC illustration of the 009 checks for the report (no new statistics). All 158 mice; per-mouse metrics from
the exploration (005) and confirmation (008) K = 2 fits."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

OUT = Path(__file__).resolve().parents[1]
A = OUT / 'artifacts'
DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
COH = {'bwm': '#2a78d6', 'lifespan': '#eb6834'}

met = pd.concat([pd.read_csv(A / '005_mouse_metrics.csv'), pd.read_csv(A / '008_mouse_metrics_confirmation.csv')],
                ignore_index=True)
dm = pd.read_parquet(A / '001_design_matrix.parquet')
times = pd.read_parquet(DATA / 'trials.parquet', columns=['session_id', 'trial_id', 'stimOn_times', 'firstMovement_times'])
t = dm.merge(times, on=['session_id', 'trial_id'])
t = t[(t.trial_id < 90) & (t.stim.abs() >= 0.25)].assign(contrast=lambda d: d.stim.abs(),
                                                         rt=lambda d: d.firstMovement_times - d.stimOn_times)
overlap = (met[met.cohort == 'lifespan'].age.min(), met[met.cohort == 'bwm'].age.max())

def draw_checks(ax):
    """Panels 1-3: occupancy and engaged weight vs age, and per-mouse state separation."""
    for a, col, ylabel in [(ax[0], 'soft_occupancy', 'soft occupancy'), (ax[1], 'engaged_stim_weight', 'engaged stimulus weight')]:
        a.axvspan(*overlap, color='#8a8985', alpha=0.15, lw=0, label='shared age window')
        for cohort, c in COH.items():
            m = met[met.cohort == cohort]
            a.scatter(m.age, m[col], color=c, s=24, edgecolor='white', lw=0.6, label=cohort)
        a.set(xlabel='mouse age (days)', ylabel=ylabel, title=f'{ylabel.capitalize()} vs age, all 158 mice')
    ax[0].legend(frameon=False, fontsize=8)

    for cohort, c in COH.items():
        m = met[met.cohort == cohort]
        ax[2].scatter(m.disengaged_stim_weight, m.engaged_stim_weight, color=c, s=24, edgecolor='white', lw=0.6, label=cohort)
    lim = [met.disengaged_stim_weight.min(), met.engaged_stim_weight.max()]
    ax[2].plot(lim, lim, color='#8a8985', ls=':', lw=1, label='equal weights')
    ax[2].axhline(5, color='#8a8985', lw=0.8, label='engaged weight 5')
    ax[2].set(xlabel='per-mouse disengaged-state stimulus weight', ylabel='per-mouse engaged-state stimulus weight',
              title='State separation per mouse')
    ax[2].legend(frameon=False, fontsize=8)


fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
draw_checks(ax)

# Check 6: reaction time on easy-trial errors, first 90 trials (pooled error trials; negative = moving before onset)
bins = np.linspace(-0.4, 0.6, 41)
for cohort, c in COH.items():
    for contrast, ls in [(1.0, '-'), (0.25, '--')]:
        r = t[(t.cohort == cohort) & (t.correct == 0) & (t.contrast == contrast)].rt
        ax[3].hist(r, bins, density=True, histtype='step', color=c, ls=ls, lw=1.8,
                   label=f'{cohort}, {int(contrast * 100)}% (n={len(r)})')
ax[3].axvline(0, color='#0b0b0b', lw=0.8)
ax[3].set(xlabel='first movement − stimulus onset (s)', ylabel='density',
          title='Reaction time on easy-trial errors, first 90 trials')
ax[3].legend(frameon=False, fontsize=8)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'confirmatory-analyses' / '010_posthoc_figure.png', dpi=110)
# report version: panels 1-3 only (replaces the earlier image crop)
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
draw_checks(ax)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'confirmatory-analyses' / '010_posthoc_figure_panels1-3.png', dpi=110)
