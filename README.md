# ibl-ai-agent-reports

Reports produced with the IBL AI agent, published with GitHub Pages from `docs/` at https://bethanjenks.github.io/ibl-ai-agent-reports/.

Each report lives in `docs/<slug>/`: the rendered `index.html`, its figures, the Quarto source `report.qmd`, and the analysis scripts in `code/` (see `code/README.md` for how to rerun them). The HTML is self-contained (figures embedded); the PNGs are kept alongside so the source can be re-rendered:

```bash
cd docs/<slug> && quarto render report.qmd --output index.html
```

The three ageing reports build on each other: the GLM-HMM engagement report reuses the RT-variability report's mouse split and RT metrics, and the quenching report uses the GLM-HMM engagement labels. Run their code in that order.

The older single-file versions in `docs/reports/` are kept so existing links keep working.

| Report | Rendered | Source | Code |
|---|---|---|---|
| Choice-selective neurons around movement onset in the unbiased block: a brain-wide map | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/choice-selectivity-movement-onset/) | [report.qmd](docs/choice-selectivity-movement-onset/report.qmd) | [code/](docs/choice-selectivity-movement-onset/code/) |
| Reaction-time variability increases with age in mice performing the IBL decision task | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/aging-rt-variability/) | [report.qmd](docs/aging-rt-variability/report.qmd) | [code/](docs/aging-rt-variability/code/) |
| Older mice spend less time engaged in the IBL decision task (builds on the RT-variability report) | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/aging-glmhmm-engagement/) | [report.qmd](docs/aging-glmhmm-engagement/report.qmd) | [code/](docs/aging-glmhmm-engagement/code/) |
| Weaker variability quenching in older mice is not explained by disengagement (builds on the GLM-HMM engagement report) | [page](https://bethanjenks.github.io/ibl-ai-agent-reports/aging-quenching-engagement/) | [report.qmd](docs/aging-quenching-engagement/report.qmd) | [code/](docs/aging-quenching-engagement/code/) |
