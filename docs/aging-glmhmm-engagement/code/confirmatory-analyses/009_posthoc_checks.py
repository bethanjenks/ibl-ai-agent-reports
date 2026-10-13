"""POST HOC / EXPLORATORY checks requested by the user after the confirmatory tests (2026-10-12).
Nothing here is a confirmatory test; it is kept separate from the locked C1-C3 results.

Per-mouse GLM-HMM metrics come from two separate K = 2 global fits: exploration mice (005) and confirmation mice
(008); each global fit pooled both cohorts, then every mouse was refit from it.
  1 age overlap        2 split balance      3 state comparability
  4 H3 (text only)     5 sessions per mouse 6 C3 vision check (errors by contrast, RT on errors)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from ibl_ai_agent.data_locations import resolve_dataset_dir

OUT = Path(__file__).resolve().parents[1]
A = OUT / 'artifacts'
DATA = resolve_dataset_dir('ibl_aging_zang2025') / 'metadata'
pd.set_option('display.width', 200)

split = pd.read_csv(A / 'split.csv')
met = pd.concat([pd.read_csv(A / '005_mouse_metrics.csv').assign(split='exploration'),
                 pd.read_csv(A / '008_mouse_metrics_confirmation.csv').assign(split='confirmation')], ignore_index=True)
bwm, life = met[met.cohort == 'bwm'], met[met.cohort == 'lifespan']

print('== 1. Age overlap')
print(life.sort_values('age')[['subject_id', 'split', 'age', 'n_sessions', 'soft_occupancy', 'engaged_stim_weight']]
      .round(2).to_string(index=False))
print(f'BWM age range: {bwm.age.min():.0f}–{bwm.age.max():.0f} d')
overlap_life = life[life.age <= bwm.age.max()]
overlap_bwm = bwm[bwm.age >= overlap_life.age.min()]           # BWM mice in the shared age window
print(f'shared window {overlap_life.age.min():.0f}–{bwm.age.max():.0f} d: '
      f'{len(overlap_life)} lifespan vs {len(overlap_bwm)} BWM mice')
print(pd.DataFrame({'lifespan (overlap)': overlap_life[['age', 'soft_occupancy', 'engaged_stim_weight']].median(),
                    'BWM (overlap)': overlap_bwm[['age', 'soft_occupancy', 'engaged_stim_weight']].median()}).round(2))
print(overlap_bwm[['subject_id', 'split', 'lab', 'age', 'soft_occupancy', 'engaged_stim_weight']].round(2).to_string(index=False))

print('\n== 2. Split balance (lifespan)')
print(life.groupby('split').age.agg(['size', 'min', 'median', 'max']).round(0))

print('\n== 3. State comparability (per-mouse refits)')
print(met.groupby('cohort')[['engaged_stim_weight', 'disengaged_stim_weight']].describe(percentiles=[.5])
      .round(2).loc[:, (slice(None), ['count', 'min', '50%', 'max'])].to_string())
met['separation'] = met.engaged_stim_weight - met.disengaged_stim_weight
print('engaged − disengaged weight, median:', met.groupby('cohort').separation.median().round(2).to_dict())
print('mice with engaged weight < 5 (global threshold):', met[met.engaged_stim_weight < 5].groupby('cohort').size().to_dict(),
      'of', met.groupby('cohort').size().to_dict())

print('\n== 5. Sessions per mouse (all sessions used, not first recording only)')
print(pd.crosstab(met.cohort, met.n_sessions))

print('\n== 6. C3 vision check: easy errors in the first 90 trials, by contrast (all 158 mice, mouse-level)')
dm = pd.read_parquet(A / '001_design_matrix.parquet')
times = pd.read_parquet(DATA / 'trials.parquet', columns=['session_id', 'trial_id', 'stimOn_times', 'firstMovement_times'])
t = dm.merge(times, on=['session_id', 'trial_id'])
t = t[(t.trial_id < 90) & (t.stim.abs() >= 0.25)].assign(contrast=lambda d: d.stim.abs(),
                                                         rt=lambda d: d.firstMovement_times - d.stimOn_times)
per = (t.groupby(['cohort', 'subject_id', 'contrast'])
       .agg(error_rate=('correct', lambda c: 1 - c.mean()), n=('correct', 'size')).reset_index())
print(per.groupby(['contrast', 'cohort']).error_rate.agg(['median', 'mean', 'size']).round(3))
err = t[t.correct == 0]
rt_err = err.groupby(['cohort', 'subject_id', 'contrast']).rt.median().reset_index()
print('median of per-mouse median reaction time on errors (s):')
print(rt_err.groupby(['contrast', 'cohort']).rt.agg(['median', 'size']).round(3))
print('pooled error trials:', err.groupby(['contrast', 'cohort']).size().to_dict())
