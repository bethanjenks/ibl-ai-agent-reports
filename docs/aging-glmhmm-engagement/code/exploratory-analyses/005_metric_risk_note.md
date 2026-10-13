# Metric risk note: K = 2 GLM-HMM with per-mouse labels (step 6b)

Global fit: engaged stimulus weight 14.5, disengaged 2.2. Figure: `005_k2_individual_fits.png`.

| metric | other global seed (ρ) | alternate sessions, ICC (n = 36) | session halves, ICC (n = 48) |
| --- | --- | --- | --- |
| occupancy (P > 0.5) | 1.00 | 0.36 | 0.46 |
| soft occupancy (mean P) | 1.00 | 0.39 | 0.50 |
| engaged stimulus weight | 1.00 | 0.76 | 0.86 |
| lapse proxy (error at 100%) | — | 0.48 | 0.28 |
| disengaged bout / switches | 1.00 | Spearman 0.05 / 0.22 | Spearman 0.43 / 0.25 |

## Supported
- K = 2 removes the fitting instability: per-mouse metrics are identical across global seeds.
- Engaged and disengaged states are separated in every mouse (per-mouse stimulus weights ~4–26 vs ~−3–6).
- Occupancy from K = 2 tracks K = 3 occupancy (ρ 0.71) where K = 3 was stable.
- Engaged stimulus weight is reliable (ICC 0.76–0.86).

## Not supported / risks
- Occupancy is only moderately reliable between sessions or session halves (ICC 0.36–0.50; median |difference|
  0.08, 90th percentile 0.27–0.30). This matches the raw lapse proxy (ICC 0.28–0.48), so engagement looks partly
  like a day-to-day state rather than a fixed trait of the mouse. Averaging over a mouse's sessions helps, but lifespan
  mice have 1–3 sessions.
- Most BWM mice sit at 0.8–1.0 (ceiling), which compresses rank correlations.
- Bout length and switch rate are unreliable; descriptive only.
- Reward history may drive disengagement (Mohammadi et al. 2025), so occupancy differences could reflect motivation.

## Recommendation
Proceed to the age analysis (step 8) with soft occupancy (no threshold, slightly more reliable) for H1 and engaged
stimulus weight as co-primary (H3); bout length and switch rate descriptive (H2 demoted). Expect limited power for H1.
