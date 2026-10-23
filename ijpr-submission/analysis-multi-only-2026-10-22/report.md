# Multi-only ablation: three training seeds

CaSiNo/DnD-only Selective τ=.4, seeds42/43/44. All **4,653 reported evaluation episodes** completed with zero execution failures.

| Opponent | Deal quality | All-episode utility | Deal rate (%) |
| --- | --- | --- | --- |
| GPT-5.4 | 0.4793 ± 0.0308 | 0.4360 ± 0.0049 | 90.01 ± 5.16 |
| GPT-5.4-mini | 0.5322 ± 0.0257 | 0.4341 ± 0.0264 | 78.21 ± 3.19 |
| Qwen3.6 native | 0.6348 ± 0.0269 | 0.5709 ± 0.0094 | 91.68 ± 3.37 |

Values are mean ± sample SD across three training seeds. Quality and utility weight the five datasets equally; non-deals contribute zero utility. Deal rate pools all episodes within each seed. Differences use unrounded values.

## Transfer against the untrained learner — GPT-5.4

| Dataset | Untrained utility | Multi-only utility | Multi-only − untrained | Joint Selective τ=.4 |
| --- | --- | --- | --- | --- |
| Amazon | 0.0913 | 0.1157 ± 0.0449 | +0.0243 | 0.4209 |
| Craigslist | 0.0234 | 0.0107 ± 0.0151 | -0.0126 | 0.3612 |
| JI | 0.5927 | 0.7122 ± 0.0203 | +0.1195 | 0.6530 |

The untrained comparison is available only against GPT-5.4.

## Comparison with joint training

| Opponent | Tasks | Joint utility | Multi-only utility | Multi-only − joint |
| --- | --- | --- | --- | --- |
| GPT-5.4 | Trained CaSiNo/DnD | 0.6562 | 0.6707 ± 0.0427 | +0.0145 |
| GPT-5.4 | Untrained price tasks | 0.3910 | 0.0632 ± 0.0282 | -0.3278 |
| GPT-5.4 | Held-out JI | 0.6530 | 0.7122 ± 0.0203 | +0.0592 |
| GPT-5.4-mini | Trained CaSiNo/DnD | 0.6781 | 0.6587 ± 0.0535 | -0.0194 |
| GPT-5.4-mini | Untrained price tasks | 0.5379 | 0.1202 ± 0.0420 | -0.4178 |
| GPT-5.4-mini | Held-out JI | 0.5553 | 0.6128 ± 0.0544 | +0.0575 |
| Qwen3.6 native | Trained CaSiNo/DnD | 0.6788 | 0.7546 ± 0.0243 | +0.0758 |
| Qwen3.6 native | Untrained price tasks | 0.8979 | 0.3633 ± 0.0311 | -0.5346 |
| Qwen3.6 native | Held-out JI | 0.5660 | 0.6189 ± 0.0408 | +0.0529 |

Joint training includes Amazon, Craigslist, CaSiNo and DnD; JI is held out in both designs. Comparisons are descriptive.

## Interpretation

- Against the untrained learner on GPT-5.4, Amazon utility improves **+.0243 on average**: seeds42 (+.0749) and43 (+.0090) improve, while seed44 declines (−.0108). Craigslist improves only for seed42 (+.0020); its three-seed mean declines by .0126.
- JI utility improves in **all three seeds**, averaging **+.1195** over untrained. Multi-only also has higher JI utility than joint training across all three opponents.
- Joint training performs better overall, with substantially stronger price-task utility across all three opponents.
- This addresses the missing multi-only ablation. Threshold-specific transfer remains unresolved without a multi-only Surplus control.
