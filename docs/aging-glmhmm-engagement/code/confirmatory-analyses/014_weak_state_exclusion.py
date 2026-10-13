"""POST HOC (user request 2026-10-13, not confirmatory): does the C1a time-engaged difference survive excluding mice
whose engaged state barely follows the stimulus (engaged weight below a threshold), where the 'engaged' label is weak?
Confirmation mice; per-mouse metrics from the locked fit (008)."""
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0, 5, 7]          # 0 = locked analysis; 5 = global engaged/disengaged threshold; 7 = stricter

m = pd.read_csv(OUT / 'artifacts' / '008_mouse_metrics_confirmation.csv')
rows = []
for thr in THRESHOLDS:
    s = m[m.engaged_stim_weight >= thr]
    life, bwm = s[s.cohort == 'lifespan'].soft_occupancy, s[s.cohort == 'bwm'].soft_occupancy
    r = mannwhitneyu(life, bwm, alternative='less')
    rows.append(dict(min_engaged_weight=thr, n_lifespan=len(life), n_bwm=len(bwm), lifespan_median=life.median(),
                     bwm_median=bwm.median(), U=r.statistic, auc=r.statistic / (len(life) * len(bwm)),
                     p_one_sided=r.pvalue))
res = pd.DataFrame(rows)
res.to_csv(OUT / 'confirmatory-analyses' / '014_weak_state_exclusion.csv', index=False)
print(res.round(4).to_string(index=False))
for cohort, g in m.groupby('cohort'):
    print(cohort, 'rho(soft occupancy, engaged weight) =', round(spearmanr(g.soft_occupancy, g.engaged_stim_weight).statistic, 2))
