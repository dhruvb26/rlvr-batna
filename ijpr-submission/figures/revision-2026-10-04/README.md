# IJPR revision figure snapshot

Twelve plots are supplied as PNG previews and vector PDFs for manuscript use.
This is the October 4, 2026 result snapshot; it does not include the later
Selective tau .2 experiments. The manuscript's existing figure references still
point to the earlier figures in the parent directory.

| Plot stem | Content |
|---|---|
| `reward_design` | Surplus and threshold terminal reward illustration at tau .4 |
| `reward_curves` | Surplus versus Selective tau .4 training reward |
| `per_dataset_br_price` | Selective tau .4 training deal quality on Amazon/Craigslist |
| `per_dataset_br_multi` | Selective tau .4 training deal quality on DnD/CaSiNo |
| `first_bid_ratio` | Opening price offer divided by buyer budget among training deals |
| `deal_walk_rate` | Selective tau .4 training deal and walk-away rates |
| `bar_results` | GPT-5.4 evaluation deal quality by dataset |
| `bar_all_episode_utility` | GPT-5.4 utility by dataset, assigning unsuccessful episodes zero |
| `bar_deal_rate` | GPT-5.4 agreement rate by dataset |
| `evaluation_datasets_gpt54` | Dataset-level GPT-5.4 comparison, including Selective tau .6 |
| `evaluation_opponents` | Surplus, Selective tau .4, and Uniform tau .4 across nine opponents |
| `tau_sensitivity` | Selective tau .4 versus .6 across three opponents |

Training curves apply a centered, edge-padded five-step moving average to each
seed. Lines show the mean across training seeds 42/43/44; shading shows one sample
standard deviation. Evaluation error bars also show one sample standard deviation
across the three trained checkpoints. The untrained baseline has one evaluation
and no seed error bar. These are not confidence intervals.

The Craigslist training-quality curve retains the logged price outlier near
update 63: one seed42 deal offered $150,000 against a $600 buyer budget. The raw
normalized utility was approximately -1494; the training reward was clipped to
-1. Smoothing spreads its effect over adjacent updates. The plotted deal-quality
series uses the raw ratio, so the outlier is not removed from this snapshot.

`figure_data.json` preserves numeric inputs for the five training plots and three
`bar_*` plots, plus the smoothing description and original source references.
It is not a complete data export for the three overview/sensitivity plots.
Its absolute paths are historical provenance from the original checkout; saved
logs and the remaining analysis outputs are not bundled in this figure commit.

The snapshot combines historical protocols: original seed42 training used v2,
while later seeds used v3. The original GPT-5.4 Surplus/Selective seed42 main
evaluations used OpenRouter/v2; Uniform seed42 and later GPT-5.4 runs used direct
OpenAI/v3. Variation across seeds therefore includes these implementation and
provider differences. The plots should not be treated as new runs under the
current source or as a full tau .2/.4/.6 comparison.


## Local regeneration

From the repository root, run:

```bash
uv run --locked python -m scripts.gen_revision_handoff_figures --output files/revision-figure-preview
```

The generator writes nine PNG/PDF pairs (the reward illustration, five training
plots, and three `bar_*` plots) plus `figure_data.json` only inside the selected
output directory. It never reads or writes the handoff document. Importing it
also does not generate files. The three overview/sensitivity plots are separate
snapshot artifacts outside this generator's original scope.

It requires the local `inventory-2026-10-04/inventory.json` and
`analysis-quality-mechanism-2026-10-04/results.json` evidence files under
`ijpr-submission/`, plus the training metrics referenced by that inventory under
`logs/`. These inputs remain outside the figure/scripts commits. Use `--root`,
`--inventory`, and `--analysis` to select another evidence location; relative
arguments resolve under `--root`. Historical absolute log paths are rebased to
the selected root. Generated provenance paths are relative to that root when
possible; this committed snapshot retains its original historical references.
