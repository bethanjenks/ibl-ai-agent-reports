"""Post-hoc (descriptive, pre-declared) variants on the confirmation set.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

Per unit, same omnibus machinery as the locked test, four variants:
  pre   : sub-bins [-0.2, -0.1, 0] s (preparation), choice labels
  post  : sub-bins [0, 0.1, 0.2] s (execution), choice labels
  cgs   : locked window, choice labels shuffled within stimulus side (choice | stimulus)
  sgc   : locked window, stimulus-side labels shuffled within choice (stimulus | choice)
Output: per session x region (>= 5 units) fraction of selective units for each variant, plus
the locked-window fraction for reference, and the per-unit table for the top regions.
"""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "exploratory-analyses"))
from common import load_tables, binned_counts, omnibus_stats, permuted_labels, SUB_EDGES, ARTIFACTS

N_PERM, ALPHA, MIN_UNITS = 1000, 0.05, 5
VARIANTS = {"locked": (SUB_EDGES, "choice", None), "pre": (SUB_EDGES[:3], "choice", None),
            "post": (SUB_EDGES[2:], "choice", None), "cgs": (SUB_EDGES, "choice", "stim"), "sgc": (SUB_EDGES, "stim", "choice")}

units, trials, split = load_tables()
rows, unit_rows, t0 = [], [], time.time()
for k, (eid, units_eid) in enumerate(units[units.eid.isin(split.eid[split.set == "confirmation"])].groupby("eid")):
    seed = int(eid.replace("-", ""), 16) % 2**32
    sig, regions, meta = {v: [] for v in VARIANTS}, [], []
    for pid in units_eid.pid.unique():
        counts, u, t = binned_counts(pid, units, trials)
        lab = {"choice": (t.side == "left").to_numpy(), "stim": (t.stim_side == "left").to_numpy()}
        strata = {"stim": t.stim_side.to_numpy(), "choice": t.side.to_numpy(), None: None}
        for v, (edges, label, stratum) in VARIANTS.items():
            labels = permuted_labels(lab[label], np.random.default_rng(seed), N_PERM, strata[stratum])
            s, _ = omnibus_stats(counts, t, labels, edges)
            thr = np.nanquantile(s[:, 1:], 1 - ALPHA, axis=1)
            sig[v].append(np.nan_to_num(s[:, 0]) > thr)
        regions.append(u.beryl_acronym.to_numpy()); meta.append(u[["pid", "cluster_id", "lab", "subject"]])
    regions = np.concatenate(regions); meta = pd.concat(meta, ignore_index=True)
    sig = {v: np.concatenate(a) for v, a in sig.items()}
    unit_rows.append(meta.assign(eid=eid, region=regions, **{f"sig_{v}": sig[v] for v in VARIANTS}))
    for reg in np.unique(regions):
        m = regions == reg
        if m.sum() >= MIN_UNITS:
            rows.append(dict(eid=eid, region=reg, lab=meta.lab.iloc[0], subject=meta.subject.iloc[0], n_units=m.sum(),
                             **{f"frac_{v}": sig[v][m].mean() for v in VARIANTS}))
    if k % 40 == 0: print(f"{k + 1} sessions, {time.time() - t0:.0f} s", flush=True)
pd.DataFrame(rows).to_csv(ARTIFACTS / "008_posthoc_session_region.csv", index=False)
pd.concat(unit_rows, ignore_index=True).to_parquet(ARTIFACTS / "008_posthoc_units.parquet")
print(f"done: {len(rows)} session-regions, {time.time() - t0:.0f} s")
