"""Confirmatory step 2: locked region tests on the confirmation set.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

Primary: session-permutation test of the mean per-session fraction of selective units, 100,000
null draws, one-sided, BH across tested regions (>= 5 eligible sessions), q < 0.05 = answer set.
Sensitivity: pooled binomial and session Wilcoxon, each BH-corrected separately.
Effect size: mean fraction with bootstrap 95% CI over sessions.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "exploratory-analyses"))
from common import region_tests, PROJECT, ARTIFACTS

N_PERM, ALPHA, MIN_SESSIONS, N_DRAWS, N_BOOT = 1000, 0.05, 5, 100_000, 2000
rng = np.random.default_rng(2026)

keys = pd.read_csv(ARTIFACTS / "006_confirmation_session_region_keys.csv")
f = np.load(ARTIFACTS / "006_confirmation_session_region_fractions.npz")["f"]
res = region_tests(keys, f, rng, N_PERM, ALPHA, MIN_SESSIONS, N_DRAWS)
# bootstrap CI of the mean session fraction
ci = {}
for reg, idx in keys.groupby("region").indices.items():
    if len(idx) >= MIN_SESSIONS:
        boots = f[rng.choice(idx, (N_BOOT, len(idx))), 0].mean(1)
        ci[reg] = np.percentile(boots, [2.5, 97.5])
res["ci_low"] = res.region.map(lambda r: ci[r][0]); res["ci_high"] = res.region.map(lambda r: ci[r][1])
res["n_subjects"] = res.region.map(keys.merge(pd.read_parquet(ARTIFACTS / "unbiased_trials.parquet")[["eid", "subject"]].drop_duplicates())
                                      .groupby("region").subject.nunique())
res["significant"] = res.q_sessperm < 0.05
res.to_csv(PROJECT / "confirmatory-analyses/007_region_tests.csv", index=False)

sig = res[res.significant]
print(f"tested regions: {len(res)}; significant (session-permutation, BH q < 0.05): {len(sig)}")
print("agreement with sensitivity tests among tested regions: binomial", int(((res.q_binom < 0.05) == res.significant).sum()),
      "/", len(res), " wilcoxon", int(((res.q_wilcoxon < 0.05) == res.significant).sum()), "/", len(res))
print(sig[["region", "n_sessions", "n_subjects", "n_units", "mean_frac", "ci_low", "ci_high", "p_sessperm", "q_sessperm", "q_binom", "q_wilcoxon"]].round(4).to_string(index=False))
print("--- not significant:")
print(res[~res.significant][["region", "n_sessions", "n_units", "mean_frac", "ci_low", "ci_high", "q_sessperm"]].round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(20, max(8, 0.22 * len(res))), gridspec_kw={"width_ratios": [3, 1]})
ax = axes[0]; y = np.arange(len(res))
colors = np.where(res.significant, "C0", "grey")
for m, col, lab in ((res.significant, "C0", "significant (BH q < 0.05)"), (~res.significant, "grey", "not significant")):
    ax.errorbar(res.mean_frac[m], y[m], xerr=[res.mean_frac[m] - res.ci_low[m], res.ci_high[m] - res.mean_frac[m]],
                fmt="o", ms=3.5, color=col, capsize=2, lw=1, label=lab)
ax.scatter(res.null_mean, y, marker="|", c="k", s=40, label="permutation null mean")
ax.axvline(ALPHA, c="grey", ls="--", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{r} ({n} sess.)" for r, n in zip(res.region, res.n_sessions)], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("fraction of choice-selective units (mean over sessions, 95% bootstrap CI)")
ax.set_title(f"Confirmation set: {len(sig)} of {len(res)} tested Beryl regions significant"); ax.legend(fontsize=8, loc="lower right")
ax = axes[1]
ax.scatter(-np.log10(res.p_sessperm), y, s=12, c=colors, label="session permutation")
ax.scatter(-np.log10(res.p_binom.clip(1e-12)), y, s=12, marker="x", c="C1", label="pooled binomial")
ax.scatter(-np.log10(res.p_wilcoxon.clip(1e-12)), y, s=12, marker="+", c="C2", label="session Wilcoxon")
ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlabel("-log10 p (uncorrected)"); ax.legend(fontsize=7); ax.set_xlim(0, 12.5)
fig.tight_layout(); fig.savefig(PROJECT / "confirmatory-analyses/007_region_tests.png", dpi=110)
