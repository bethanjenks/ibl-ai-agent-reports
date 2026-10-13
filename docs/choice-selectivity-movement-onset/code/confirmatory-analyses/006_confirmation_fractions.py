"""Confirmatory step 1: locked per-unit statistic and session x region fractions on the confirmation set.

Original question: Using only trials from the initial unbiased block, which brain regions
contain neurons whose firing rate around movement onset differs significantly between
left and right choices?

Locked plan (question.md): window [-0.2, 0.2] s around firstMovement_times, 4 x 100 ms sub-bins,
pre-stimulus baseline subtraction, omnibus sum of squared standardized differences, 1000 shared
label permutations per session seeded from the session UUID, unit selective if above its own
95th percentile; session x region needs >= 5 units. Output: fraction of selective units per
session x region under the true labels and under each permutation.
"""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "exploratory-analyses"))
from common import load_tables, session_region_fractions, SUB_EDGES, ARTIFACTS

N_PERM, ALPHA, MIN_UNITS = 1000, 0.05, 5

units, trials, split = load_tables()
conf = split.eid[split.set == "confirmation"]
t0 = time.time()
keys, f = session_region_fractions(units, trials, conf, SUB_EDGES, N_PERM, ALPHA, MIN_UNITS)
print(f"{keys.eid.nunique()} sessions, {len(keys)} session-regions, {keys.n_units.sum()} units, {time.time() - t0:.0f} s")
np.savez_compressed(ARTIFACTS / "006_confirmation_session_region_fractions.npz", f=f)
keys.to_csv(ARTIFACTS / "006_confirmation_session_region_keys.csv", index=False)
