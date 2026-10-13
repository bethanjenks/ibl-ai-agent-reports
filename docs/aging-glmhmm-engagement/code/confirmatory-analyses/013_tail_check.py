"""POST HOC (user request 2026-10-12): which lifespan mice stay inaccurate on early easy trials after removing trials
with first movement before stimulus onset, and are they the mice with weak engaged states (engaged weight < 5)?
Confirmation mice; engaged weight from the locked fit (008), filtered accuracy from 012 (premove_out)."""
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

A = Path(__file__).resolve().parents[1] / 'artifacts'
TAIL = 0.87                     # accuracy below this = tail (lifespan values 0.76-0.86 vs the rest >= 0.92)
base = pd.read_csv(A / '008_mouse_metrics_confirmation.csv')[['subject_id', 'cohort', 'age', 'engaged_stim_weight']]
filt = pd.read_csv(A / '012_metrics_premove_out.csv')[['subject_id', 'acc_easy_early']].rename(
    columns={'acc_easy_early': 'acc_easy_early_filtered'})
m = base.merge(filt, on='subject_id').assign(in_tail=lambda d: d.acc_easy_early_filtered < TAIL,
                                             weak_engaged=lambda d: d.engaged_stim_weight < 5)
m.to_csv(A / '013_tail_check.csv', index=False)
print(pd.crosstab([m.cohort, m.in_tail], m.weak_engaged))
for cohort, g in m.groupby('cohort'):
    print(cohort, 'rho(filtered accuracy, engaged weight)', round(spearmanr(g.acc_easy_early_filtered, g.engaged_stim_weight).statistic, 2),
          'rho(filtered accuracy, age)', round(spearmanr(g.acc_easy_early_filtered, g.age).statistic, 2))
