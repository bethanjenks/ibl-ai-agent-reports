"""Region-level test design on the exploration set: one p-value per region, calibration, power.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

Per session x region we have the fraction f of units with omnibus p < 0.05. Three region tests
of H0 "no unit in the region is choice-selective" (f = 0.05 in every session):
  binomial   : pooled count of significant units vs Binomial(n_units, 0.05); ignores that units
               of one session share trials, so anti-conservative.
  wilcoxon   : one-sided Wilcoxon signed-rank of per-session f against 0.05; session = replicate.
  sessperm   : observed mean over sessions of f, against a null built by drawing, for each
               session independently, one of its label permutations and recomputing f with the
               same per-unit 95th-percentile thresholds. Respects within-session correlation.
The pre-movement control window [-0.6, -0.2] s, where unit-level selectivity is at chance
(step 003), checks calibration. Power at confirmation-set session counts is estimated by
bootstrapping exploration sessions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_tables, session_region_fractions, region_tests, SUB_EDGES, PROJECT, ARTIFACTS

CONTROL_EDGES = np.round(np.arange(-0.6, -0.15, 0.1), 2)     # 4 x 100 ms, pre-movement
N_PERM = 1000
ALPHA = 0.05
MIN_UNITS = 5
MIN_SESSIONS = 5
N_DRAWS = 10000
rng = np.random.default_rng(3)

units, trials, split = load_tables()
expl = split.eid[split.set == "exploration"]
keys, f = session_region_fractions(units, trials, expl, SUB_EDGES, N_PERM, ALPHA, MIN_UNITS)
keys_c, f_c = session_region_fractions(units, trials, expl, CONTROL_EDGES, N_PERM, ALPHA, MIN_UNITS)
np.savez_compressed(ARTIFACTS / "005_exploration_session_region_fractions.npz", f=f, f_control=f_c)
keys.to_csv(ARTIFACTS / "005_exploration_session_region_keys.csv", index=False)
res = region_tests(keys, f, rng, N_PERM, ALPHA, MIN_SESSIONS, N_DRAWS)
res_c = region_tests(keys_c, f_c, rng, N_PERM, ALPHA, MIN_SESSIONS, N_DRAWS)
res.to_csv(PROJECT / "exploratory-analyses/005_region_tests_exploration.csv", index=False)
res_c.to_csv(PROJECT / "exploratory-analyses/005_region_tests_control_window.csv", index=False)

print(f"{len(res)} regions with >= {MIN_SESSIONS} exploration sessions")
for name, r in (("analysis window", res), ("control window", res_c)):
    print(name, "regions with BH q < 0.05:", {c: int((r[c] < 0.05).sum()) for c in ("q_binom", "q_wilcoxon", "q_sessperm")})
print(res[["region", "n_sessions", "n_units", "mean_frac", "null_mean", "p_binom", "p_wilcoxon", "p_sessperm"]].round(4).to_string(index=False))

# ---- power: bootstrap exploration sessions to confirmation-set session counts (sessperm test)
conf_sessions = pd.read_csv(PROJECT / "exploratory-analyses/000_region_coverage.csv", index_col=0)["n_sessions_confirmation"]
power_rows = []
for reg in res.region:
    idx = keys.index[keys.region == reg].to_numpy()
    for n_boot in sorted({5, 10, 20, int(conf_sessions.get(reg, 0))} - {0}):
        hits = 0
        for _ in range(200):
            b = rng.choice(idx, n_boot, replace=True)
            null = np.stack([f[i, rng.integers(1, N_PERM + 1, 2000)] for i in b]).mean(0)
            hits += ((null >= f[b, 0].mean()).sum() + 1) / 2001 < 0.05 / len(res)      # Bonferroni-strict proxy for BH
        power_rows.append(dict(region=reg, n_sessions=n_boot, power=hits / 200, is_conf=(n_boot == conf_sessions.get(reg, 0))))
power = pd.DataFrame(power_rows)
print("estimated power at confirmation-set session counts (alpha 0.05 / n_regions):")
print(power[power.is_conf].merge(res[["region", "mean_frac"]]).round(3).to_string(index=False))

# ---- figure
fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
ax = axes[0]; y = np.arange(len(res))
ax.errorbar(res.mean_frac, y, xerr=res.sem_frac, fmt="o", ms=4, capsize=2, label="observed mean fraction +/- SEM")
ax.scatter(res.null_mean, y, marker="|", c="k", s=60, label="session-permutation null mean")
ax.axvline(ALPHA, c="grey", ls="--", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([f"{r} ({n})" for r, n in zip(res.region, res.n_sessions)], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("fraction of units selective (session-averaged)"); ax.legend(fontsize=8); ax.set_title("Effect size per region (n sessions)")
ax = axes[1]
for c, lab in (("p_binom", "pooled binomial"), ("p_wilcoxon", "session Wilcoxon"), ("p_sessperm", "session permutation")):
    ax.scatter(-np.log10(res[c].clip(1e-5)), y, s=18, label=lab)
ax.axvline(-np.log10(0.05), c="grey", ls="--", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(res.region, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("-log10 p (clipped at 5)"); ax.legend(fontsize=8); ax.set_title("Three region tests, analysis window")
ax = axes[2]
for c, lab in (("p_binom", "pooled binomial"), ("p_wilcoxon", "session Wilcoxon"), ("p_sessperm", "session permutation")):
    ax.hist(res_c[c], np.linspace(0, 1, 11), alpha=0.5, label=f"{lab}: {(res_c[c] < 0.05).sum()} of {len(res_c)} p < 0.05")
ax.set_xlabel("p-value"); ax.set_ylabel("regions"); ax.legend(fontsize=8)
ax.set_title("Calibration: control window [-0.6, -0.2] s\n(unit-level selectivity at chance)")
ax = axes[3]
for reg in res.region.head(6).tolist() + res.region.tail(3).tolist():
    g = power[(power.region == reg) & ~power.is_conf].sort_values("n_sessions")
    ax.plot(g.n_sessions, g.power, marker="o", label=f"{reg} (f={res.set_index('region').mean_frac[reg]:.2f})")
ax.set_xlabel("sessions in confirmation set (bootstrap)"); ax.set_ylabel("power, session-permutation test"); ax.legend(fontsize=7, ncol=2)
ax.set_title("Power vs number of sessions"); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(PROJECT / "exploratory-analyses/005_region_test_design.png", dpi=110)
