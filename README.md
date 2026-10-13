# ibl-ai-agent-reports

Reports produced with the IBL AI agent, published with GitHub Pages from `docs/` at https://bethanjenks.github.io/ibl-ai-agent-reports/.

Each report lives in `docs/<slug>/`: the rendered `index.html`, its figures, and the Quarto source `report.qmd`. The HTML is self-contained (figures embedded); the PNGs are kept alongside so the source can be re-rendered:

```bash
cd docs/<slug> && quarto render report.qmd --output index.html
```

The older single-file versions in `docs/reports/` are kept so existing links keep working.

| Report | Rendered | Source |
|---|---|---|
| Choice-selective neurons around movement onset in the unbiased block: a brain-wide map | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/choice-selectivity-movement-onset/) | [report.qmd](docs/choice-selectivity-movement-onset/report.qmd) |
| Reaction-time variability increases with age in mice performing the IBL decision task | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/aging-rt-variability/) | [report.qmd](docs/aging-rt-variability/report.qmd) |
| Older mice spend less time engaged in the IBL decision task | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/aging-glmhmm-engagement/) | [report.qmd](docs/aging-glmhmm-engagement/report.qmd) |
