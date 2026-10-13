"""Post-hoc checks added at report review (descriptive, no tests; confirmation mice).

1. Lower RT bound: does removing fast responses (< 0.15 or < 0.2 s) remove the sd_log_rt age effect?
2. S4 effect size: bootstrap 95% CI over BWM mice for ρ(age, median RT).
3. Paper's session QC (> 400 trials, ≥ 90% correct at 100% contrast, ≥ 3 errors after trial filtering):
   do P1, S1, S3 and median RT change when it is applied?
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / 'exploratory-analyses'))
from rt_metrics import session_metrics, mouse_metrics

trials = pd.read_parquet(OUT / 'artifacts' / '005_trials_rt_confirmation.parquet')
rho = lambda d, k: stats.spearmanr(d.age_at_session_days, d[k])[0]

rows = []
for window in [(0.08, 2.0), (0.15, 2.0), (0.2, 2.0)]:
    m = mouse_metrics(session_metrics(trials, window=window))
    b = m[m.cohort == 'bwm']
    rows.append(dict(check='lower bound', setting=f'{window[0]:g}–{window[1]:g} s', mice=len(m),
                     sd_log_rt_all=rho(m, 'sd_log_rt'), sd_log_rt_bwm=rho(b, 'sd_log_rt'), cv_all=rho(m, 'cv'),
                     median_rt_all=rho(m, 'median_rt')))

# Paper's session QC, on the paper's trial rules (first 400, complete, RT 0.08–2 s)
n_trials = trials.groupby('session_id').size()
kept = trials[(trials.trial_id < 400) & trials.complete & trials.rt.between(0.08, 2.0)]
acc_full = kept[kept.contrast == 1].groupby('session_id').feedbackType.apply(lambda x: np.mean(x == 1))
n_err = kept.groupby('session_id').feedbackType.apply(lambda x: np.sum(x == -1))
qc_pass = ((n_trials > 400) & (acc_full >= 0.9) & (n_err >= 3)).fillna(False)
print(f'paper session QC: {qc_pass.sum()} / {len(qc_pass)} confirmation sessions pass')
s = session_metrics(trials)
m = mouse_metrics(s[s.session_id.isin(qc_pass[qc_pass].index)])
b = m[m.cohort == 'bwm']
rows.append(dict(check='paper session QC', setting=f'{qc_pass.sum()}/{len(qc_pass)} sessions', mice=len(m),
                 sd_log_rt_all=rho(m, 'sd_log_rt'), sd_log_rt_bwm=rho(b, 'sd_log_rt'), cv_all=rho(m, 'cv'),
                 median_rt_all=rho(m, 'median_rt')))
res = pd.DataFrame(rows)
res.to_csv(OUT / 'confirmatory-analyses' / '007_review_checks.csv', index=False)
print(res.round(2).to_string())

# Bootstrap CI for S4 (resample BWM mice)
mice = pd.read_csv(OUT / 'artifacts' / '005_mouse_metrics_confirmation.csv')
bwm = mice[mice.cohort == 'bwm']
rng = np.random.default_rng(0)
boot = [rho(bwm.iloc[rng.integers(0, len(bwm), len(bwm))], 'median_rt') for _ in range(5000)]
print(f"S4 ρ(age, median RT), BWM: {rho(bwm, 'median_rt'):+.2f}, 95% bootstrap CI {np.percentile(boot, [2.5, 97.5]).round(2)}")
